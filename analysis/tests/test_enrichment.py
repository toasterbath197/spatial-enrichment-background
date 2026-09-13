"""
Tests for the enrichment engine.

WHY THESE EXIST: the central claim of this project is that a p-value changes when
you change the background gene set. That claim is only as trustworthy as the
function computing the p-value. These tests check that function against cases you
can verify by hand or against scipy directly, without running the pipeline.

Every test is written so a reader can confirm the expected value themselves.
Where a number is asserted, the comment says where it comes from.

RUN:  python3 -m pytest analysis/tests/ -v
      (or: python3 analysis/tests/test_enrichment.py  for a plain-python run)
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.stats import hypergeom

from de_pipeline import enrichment as en


# --------------------------------------------------------------------------
# Test fixtures: a tiny pathway database small enough to reason about by hand.
#
#   Background:  10 genes, A through J
#   PATH_HIT:    5 genes (A B C D E)  - the query overlaps this heavily
#   PATH_MISS:   5 genes (F G H I J)  - the query does not overlap this at all
#   Query:       4 genes (A B C D)
#
# So for PATH_HIT:  N=10, n=5, K=4, k=4   (all four query genes are in it)
#    for PATH_MISS: N=10, n=5, K=4, k=0
# --------------------------------------------------------------------------
TINY_GMT = {
    "PATH_HIT":  ("pathway the query hits",  frozenset("ABCDE")),
    "PATH_MISS": ("pathway the query misses", frozenset("FGHIJ")),
}
TINY_BACKGROUND = list("ABCDEFGHIJ")
TINY_QUERY = list("ABCD")


def _row(df, term_id):
    """Pull one term's row out of the results frame."""
    match = df[df["term_id"] == term_id]
    assert len(match) == 1, f"expected exactly one row for {term_id}, got {len(match)}"
    return match.iloc[0]


# --------------------------------------------------------------------------
# 1. The counts the test extracts must be the ones we think they are.
#    If this fails, every other test is meaningless.
# --------------------------------------------------------------------------

def test_counts_are_what_we_expect():
    df = en.hypergeometric_enrichment(
        TINY_QUERY, TINY_BACKGROUND, TINY_GMT, min_term_size=1, max_term_size=100)

    hit = _row(df, "PATH_HIT")
    assert hit["background_size"] == 10      # N: all ten background genes are annotated
    assert hit["term_size_in_background"] == 5   # n: A B C D E
    assert hit["n_de_in_background"] == 4     # K: A B C D
    assert hit["overlap"] == 4                # k: all four query genes are in PATH_HIT

    miss = _row(df, "PATH_MISS")
    assert miss["overlap"] == 0               # k: no query gene is in F G H I J


# --------------------------------------------------------------------------
# 2. The p-value must match scipy computed independently.
#    We are checking the engine wires the arguments up in the right order -
#    hypergeom.sf(k-1, N, n, K) is easy to get subtly wrong.
# --------------------------------------------------------------------------

def test_pvalue_matches_scipy_directly():
    df = en.hypergeometric_enrichment(
        TINY_QUERY, TINY_BACKGROUND, TINY_GMT, min_term_size=1, max_term_size=100)
    hit = _row(df, "PATH_HIT")

    # computed here from scratch, not from the engine
    expected_p = hypergeom.sf(4 - 1, 10, 5, 4)
    assert np.isclose(hit["pvalue"], expected_p)

    # and by hand: P(all 4 of 4 draws land in the 5-gene pathway)
    #   = C(5,4)*C(5,0) / C(10,4) = 5 / 210 = 0.023809...
    assert np.isclose(hit["pvalue"], 5 / 210)


def test_a_pathway_with_no_overlap_is_not_significant():
    df = en.hypergeometric_enrichment(
        TINY_QUERY, TINY_BACKGROUND, TINY_GMT, min_term_size=1, max_term_size=100)
    miss = _row(df, "PATH_MISS")
    # P(X >= 0) is 1 by definition - drawing at least zero is certain
    assert np.isclose(miss["pvalue"], 1.0)


# --------------------------------------------------------------------------
# 3. THE CENTRAL CLAIM OF THE PAPER, as a test.
#    Hold the query, the pathway and the overlap fixed; enlarge the background
#    only. The p-value must fall. If this test fails, the paper is wrong.
# --------------------------------------------------------------------------

def test_enlarging_the_background_lowers_the_pvalue():
    pathway = frozenset(f"g{i}" for i in range(50))         # 50-gene pathway
    query = [f"g{i}" for i in range(15)]                     # 15 of them are "DE"

    pvalues = []
    for background_size in (100, 500, 2000, 10000):
        # background = the pathway plus filler genes out to the target size
        filler = [f"filler{i}" for i in range(background_size - len(pathway))]
        background = list(pathway) + filler

        # the filler genes need to be annotated somewhere, or the engine
        # discards them when it intersects against the GMT universe
        gmt = {
            "PATHWAY": ("the pathway under test", pathway),
            "FILLER_TERM": ("bookkeeping only, holds the filler genes", frozenset(filler)),
        }
        df = en.hypergeometric_enrichment(
            query, background, gmt, min_term_size=1, max_term_size=10**9)
        pvalues.append(float(_row(df, "PATHWAY")["pvalue"]))

    # strictly decreasing: each larger background gives a smaller p-value
    assert all(a > b for a, b in zip(pvalues, pvalues[1:])), (
        f"p-values should fall as background grows, got {pvalues}")

    # and the effect is large, not marginal
    assert pvalues[0] / pvalues[-1] > 1e6


def test_counts_stay_fixed_while_background_grows():
    """Companion to the test above: confirms the p-value moved because N moved,
    and NOT because the overlap or pathway size changed underneath us."""
    pathway = frozenset(f"g{i}" for i in range(50))
    query = [f"g{i}" for i in range(15)]

    seen = []
    for background_size in (100, 500, 2000):
        filler = [f"filler{i}" for i in range(background_size - len(pathway))]
        gmt = {
            "PATHWAY": ("the pathway under test", pathway),
            "FILLER_TERM": ("bookkeeping only", frozenset(filler)),
        }
        df = en.hypergeometric_enrichment(
            query, list(pathway) + filler, gmt, min_term_size=1, max_term_size=10**9)
        row = _row(df, "PATHWAY")
        seen.append((row["overlap"], row["term_size_in_background"], row["n_de_in_background"]))

    assert len(set(seen)) == 1, f"k, n and K should be identical at every N, got {seen}"
    assert seen[0] == (15, 50, 15)


# --------------------------------------------------------------------------
# 4. Fold enrichment, which the write-up reports alongside p-values.
# --------------------------------------------------------------------------

def test_fold_enrichment_is_observed_over_expected():
    df = en.hypergeometric_enrichment(
        TINY_QUERY, TINY_BACKGROUND, TINY_GMT, min_term_size=1, max_term_size=100)
    hit = _row(df, "PATH_HIT")

    # expected overlap = n * K / N = 5 * 4 / 10 = 2
    assert np.isclose(hit["expected_overlap"], 2.0)
    # fold = observed / expected = 4 / 2 = 2
    assert np.isclose(hit["fold_enrichment"], 2.0)


# --------------------------------------------------------------------------
# 5. Filtering and edge-case behaviour that the Methods section claims.
# --------------------------------------------------------------------------

def test_term_size_filter_uses_size_within_background_not_raw_gmt_size():
    """Methods §2.4 claims pathway size is judged inside the background under
    test, not against the whole database. This checks that is true, because it
    is the mechanism by which background choice changes which terms are tested."""
    big_pathway = frozenset(f"g{i}" for i in range(100))
    gmt = {"BIG": ("100 genes in the database", big_pathway)}

    # background containing only 3 of the pathway's genes
    narrow_background = ["g0", "g1", "g2", "other1", "other2"]
    df = en.hypergeometric_enrichment(
        ["g0"], narrow_background, gmt, min_term_size=5, max_term_size=500)

    # raw size is 100 (inside 5..500) but size within this background is 3,
    # which is below the minimum - so the term must be dropped
    assert len(df) == 0


def test_genes_outside_the_background_are_ignored():
    """A DE gene that isn't in the chosen background cannot be scored against it."""
    df = en.hypergeometric_enrichment(
        list("ABCD") + ["NOT_IN_BACKGROUND"], TINY_BACKGROUND, TINY_GMT,
        min_term_size=1, max_term_size=100)
    # K is still 4, not 5
    assert _row(df, "PATH_HIT")["n_de_in_background"] == 4


def test_unannotated_background_genes_do_not_change_the_result():
    """Genes with no pathway annotation are dropped before testing. This is why
    the sensitivity analysis in Fix 10 had to recompute the test directly rather
    than passing a padded background - see run_fix10_sensitivity.py."""
    padded = TINY_BACKGROUND + [f"unannotated{i}" for i in range(1000)]

    plain = en.hypergeometric_enrichment(
        TINY_QUERY, TINY_BACKGROUND, TINY_GMT, min_term_size=1, max_term_size=100)
    with_padding = en.hypergeometric_enrichment(
        TINY_QUERY, padded, TINY_GMT, min_term_size=1, max_term_size=100)

    assert float(_row(plain, "PATH_HIT")["pvalue"]) == float(_row(with_padding, "PATH_HIT")["pvalue"])
    assert _row(with_padding, "PATH_HIT")["background_size"] == 10


def test_empty_result_has_the_expected_columns():
    """Downstream code reads these column names, so an empty frame must still
    carry them rather than being a bare DataFrame()."""
    df = en.hypergeometric_enrichment(
        ["A"], ["A", "B"], {"T": ("t", frozenset("AB"))},
        min_term_size=50, max_term_size=100)  # nothing can pass this filter
    assert len(df) == 0
    for column in ("term_id", "pvalue", "padj", "fold_enrichment", "overlap"):
        assert column in df.columns


# --------------------------------------------------------------------------
# 6. Multiple-testing correction.
# --------------------------------------------------------------------------

def test_adjusted_pvalues_are_never_smaller_than_raw():
    """Benjamini-Hochberg can only raise a p-value, never lower it."""
    pathway_genes = [frozenset(f"g{i}" for i in range(start, start + 20))
                     for start in range(0, 200, 20)]
    gmt = {f"T{i}": (f"term {i}", genes) for i, genes in enumerate(pathway_genes)}
    background = [f"g{i}" for i in range(200)]
    query = [f"g{i}" for i in range(30)]

    df = en.hypergeometric_enrichment(query, background, gmt,
                                      min_term_size=1, max_term_size=10**9)
    assert (df["padj"] >= df["pvalue"] - 1e-12).all()


# --------------------------------------------------------------------------
# 7. Helpers used across the pipeline.
# --------------------------------------------------------------------------

def test_species_key_resolves_the_free_text_species_field():
    assert en.species_key("Mouse (5xFAD)") == "mouse"
    assert en.species_key("Human") == "human"
    assert en.species_key("Mouse (AppNL-G-F KI)") == "mouse"


def test_species_key_refuses_to_guess_when_ambiguous():
    """Failing loudly matters here: silently picking the wrong species would
    mean silently using the wrong pathway database."""
    for ambiguous in ("Rat", "human and mouse co-culture", ""):
        try:
            en.species_key(ambiguous)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {ambiguous!r}")


def test_parse_gmt_reads_the_expected_shape(tmp_path=None):
    import tempfile, os
    handle, path = tempfile.mkstemp(suffix=".gmt", text=True)
    with os.fdopen(handle, "w") as f:
        f.write("# a comment line, should be skipped\n")
        f.write("TERM1\tfirst description\tGENE_A\tGENE_B\n")
        f.write("TERM2\tsecond description\tGENE_B\tGENE_C\tGENE_D\n")
        f.write("\n")                            # blank line, should be skipped
        f.write("TERM_EMPTY\tno genes at all\n")  # no genes, should be skipped
    try:
        gmt = en.parse_gmt(path)
    finally:
        os.unlink(path)

    assert set(gmt) == {"TERM1", "TERM2"}
    assert gmt["TERM1"] == ("first description", frozenset({"GENE_A", "GENE_B"}))
    assert len(gmt["TERM2"][1]) == 3


# --------------------------------------------------------------------------
# 8. The real pathway databases, if present. Skipped gracefully if not.
# --------------------------------------------------------------------------

def test_real_reactome_database_has_expected_shape():
    try:
        gmt = en.load_gmt("mouse", "reactome")
        universe = en.gmt_universe("mouse", "reactome")
    except FileNotFoundError:
        print("  (skipped: Reactome GMT not present)")
        return

    assert len(gmt) == 1651, f"expected 1,651 mouse Reactome pathways, got {len(gmt)}"
    assert len(universe) == 9198, f"expected a 9,198-gene universe, got {len(universe)}"

    # a sanity check on content: Reactome's "Immune System" is a large pathway.
    # If a future database swap silently produced tiny terms - the failure mode
    # that invalidated the original GO BP results - this catches it.
    sizes = sorted(len(genes) for _desc, genes in gmt.values())
    median_size = sizes[len(sizes) // 2]
    assert median_size >= 10, (
        f"median pathway size is {median_size}; suspiciously small, check the database")


if __name__ == "__main__":
    # plain-python runner so the tests work without pytest installed
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures.append((name, exc))
            print(f"  FAIL  {name}\n        {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"  ERROR {name}\n        {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
