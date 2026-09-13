"""
Pathway over-representation analysis.

The question this module exists to answer: given a list of differentially
expressed genes, which biological pathways contain more of them than you would
expect by chance? The answer depends on what "by chance" means, and that is set
by the background gene set — which is the variable this project studies.

The core call is:

    gmt = load_gmt("mouse")                       # pathway definitions
    results = hypergeometric_enrichment(
        de_genes,          # the genes you found interesting
        background_genes,  # what you are comparing them against
        gmt,
    )

`results` is one row per pathway, with a p-value and a fold enrichment.

Two pathway databases are available. Reactome is the default. GO Biological
Process is retained only so that pre-2026-08-10 results remain reproducible;
its source files have a construction defect (see PATHWAY_DBS below) and it must
not be used for new analyses.

Tests: analysis/tests/test_enrichment.py
Method details and rationale: 00_PROJECT_NOTES_AND_METHODOLOGY.txt, sections 21-23
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

from de_pipeline.paths import PATHWAY_DB_DIR

# Both databases come from the same source repository (ELTEbioinformatics/
# GMT_files_for_mulea, commit 601ed08c) but are built by different scripts, and
# only one of those scripts is correct.
#
# GO BP IS DEFECTIVE — DO NOT USE FOR NEW RESULTS. Its build script keeps one
# representative gene per term rather than all genes annotated to it, so a
# term's apparent size counts descendant terms rather than genes. 6,138 of
# 14,158 mouse terms end up holding a single gene, and the resulting 4,852-gene
# "universe" is an artifact of that bug rather than a property of GO. Full
# diagnosis in 00_PROJECT_NOTES_AND_METHODOLOGY.txt, section 21.
PATHWAY_DBS = {
    "reactome": {
        "mouse": PATHWAY_DB_DIR / "Reactome_Mus_musculus_GeneSymbol.gmt",
        "human": PATHWAY_DB_DIR / "Reactome_Homo_sapiens_GeneSymbol.gmt",
    },
    "go_bp": {
        "mouse": PATHWAY_DB_DIR / "GO_BP_Mus_musculus_GeneSymbol.gmt",
        "human": PATHWAY_DB_DIR / "GO_BP_Homo_sapiens_GeneSymbol.gmt",
    },
    # Independent replication database (added to check whether the baseline and
    # negative-control results are an artifact of Reactome specifically, rather
    # than of background choice). Same source repository and commit as Reactome
    # (ELTEbioinformatics/GMT_files_for_mulea, 601ed08c), built by a simple
    # pass-through of WikiPathways' own official GMT (Martens et al. 2021,
    # PMID 33211851) - not a custom gene-collapsing script, so it does not share
    # whatever caused the GO BP defect. See 00_PROJECT_NOTES_AND_METHODOLOGY.txt
    # for the vetting stats (term-size distribution, single-gene-term rate).
    "wikipathways": {
        "mouse": PATHWAY_DB_DIR / "WikiPathways_Mus_musculus_GeneSymbol.gmt",
        "human": PATHWAY_DB_DIR / "WikiPathways_Homo_sapiens_GeneSymbol.gmt",
    },
}

DEFAULT_DB = "reactome"

# Pathways smaller than this can't reach significance under any plausible
# overlap; pathways larger than this are too broad to be informative. Size is
# always judged within the background under test, not against the whole
# database — see _pathway_sizes_within_background().
MIN_TERM_SIZE = 5
MAX_TERM_SIZE = 500

RESULT_COLUMNS = [
    "term_id", "description", "term_size_in_background", "overlap",
    "n_de_in_background", "background_size", "expected_overlap",
    "fold_enrichment", "pvalue", "padj",
]


# ---------------------------------------------------------------------------
# Loading pathway definitions
# ---------------------------------------------------------------------------

def parse_gmt(path) -> dict:
    """Read a GMT file into {term_id: (description, frozenset_of_genes)}.

    GMT is one pathway per line, tab-separated:
        term_id <TAB> description <TAB> gene1 <TAB> gene2 <TAB> ...

    Comment lines, blank lines and pathways with no genes are skipped.
    """
    terms = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 3:
                continue
            term_id, description = fields[0], fields[1]
            genes = frozenset(g.strip() for g in fields[2:] if g.strip())
            if genes:
                terms[term_id] = (description, genes)
    return terms


@lru_cache(maxsize=8)
def load_gmt(species: str, db: str = DEFAULT_DB) -> dict:
    """Pathway definitions for one species. Cached — the driver scripts call
    this once per gene list and the files are 1-2 MB.

    species: 'mouse' or 'human'
    db:      'reactome' (default) or 'go_bp' (defective; see PATHWAY_DBS)
    """
    if db not in PATHWAY_DBS:
        raise ValueError(f"Unknown pathway db {db!r}; expected one of {sorted(PATHWAY_DBS)}")
    return parse_gmt(PATHWAY_DBS[db][species])


@lru_cache(maxsize=8)
def gmt_universe(species: str, db: str = DEFAULT_DB) -> frozenset:
    """Every gene carrying at least one annotation in the database.

    This is what this project means by "genome-wide background", and it is the
    same convention enrichment tools use for their default background. It is
    NOT the whole genome: 9,198 genes for mouse and 10,714 for human under
    Reactome, against roughly 20,000 protein-coding genes.

    That distinction is deliberate and conservative — a fuller background makes
    the inflation this project measures larger, not smaller (quantified by
    `run_experiments.py sensitivity`).
    """
    return frozenset().union(*(genes for _description, genes in load_gmt(species, db).values()))


def species_key(species_str: str) -> str:
    """Map a free-text species field ('Mouse (5xFAD)', 'Human') to 'mouse' or
    'human'. Raises if neither or both appear — guessing here would mean
    silently using the wrong species' pathway database.
    """
    text = species_str.lower()
    has_mouse, has_human = "mouse" in text, "human" in text
    if has_mouse and not has_human:
        return "mouse"
    if has_human and not has_mouse:
        return "human"
    raise ValueError(f"Cannot resolve species from {species_str!r}")


# ---------------------------------------------------------------------------
# The enrichment test itself
#
# Broken into three steps so each can be read and checked on its own:
#   1. restrict the query and background to genes the database knows about
#   2. count, for every pathway, how big it is and how much the query overlaps
#   3. turn those counts into p-values
# ---------------------------------------------------------------------------

def _restrict_to_annotated(de_genes, background_genes, gmt):
    """Step 1. Drop genes the pathway database has never heard of, then drop
    query genes that aren't in the background.

    Genes with no annotation cannot contribute to any pathway's overlap, so
    including them would only inflate the background size without changing
    which pathways come out significant. Every enrichment tool does this.

    Returns (background, query) as frozensets, with query ⊆ background.
    """
    annotated = frozenset().union(*(genes for _description, genes in gmt.values())) if gmt else frozenset()
    background = frozenset(background_genes) & annotated
    query = frozenset(de_genes) & background
    return background, query


def _pathway_sizes_within_background(gmt, background, query,
                                     min_term_size, max_term_size):
    """Step 2. For each pathway, count its size and its overlap with the query,
    both measured *inside the given background*.

    Measuring within the background rather than against the whole database is
    the crux of this project: the same pathway is a different size depending on
    what you are comparing against, which is how background choice changes the
    answer.

    Pathways outside the size filter are dropped before p-values are computed,
    so they don't consume multiple-testing budget.

    Returns four parallel lists: term_ids, descriptions, overlaps (k), sizes (n).
    """
    term_ids, descriptions, overlaps, sizes = [], [], [], []
    for term_id, (description, genes) in gmt.items():
        genes_in_background = genes & background
        size = len(genes_in_background)
        if size < min_term_size or size > max_term_size:
            continue
        term_ids.append(term_id)
        descriptions.append(description)
        overlaps.append(len(genes_in_background & query))
        sizes.append(size)
    return term_ids, descriptions, overlaps, sizes


def _score(overlaps, sizes, n_background, n_query):
    """Step 3. Turn the counts into p-values, effect sizes and adjusted p-values.

    For one pathway, writing
        N = n_background   genes to draw from
        n = size           how many of those belong to the pathway
        K = n_query        how many genes we drew (the DE list)
        k = overlap        how many of those landed in the pathway

    the probability of seeing at least k by chance is the upper tail of the
    hypergeometric distribution, P(X >= k) = hypergeom.sf(k-1, N, n, K).

    Note where N sits: raising it lowers the expected overlap nK/N, which
    lowers the p-value for an unchanged k. That is the entire mechanism this
    project measures.

    Computed for all pathways at once, then Benjamini-Hochberg corrected across
    them.
    """
    overlaps = np.asarray(overlaps)
    sizes = np.asarray(sizes)

    pvalues = hypergeom.sf(overlaps - 1, n_background, sizes, n_query)

    expected = sizes * (n_query / n_background) if n_background else np.zeros_like(sizes, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        fold_enrichment = np.where(expected > 0, overlaps / expected, np.nan)

    padj = multipletests(pvalues, method="fdr_bh")[1] if len(pvalues) else pvalues
    return pvalues, padj, expected, fold_enrichment


def hypergeometric_enrichment(
    de_genes,
    background_genes,
    gmt: dict,
    min_term_size: int = MIN_TERM_SIZE,
    max_term_size: int = MAX_TERM_SIZE,
) -> pd.DataFrame:
    """Test every pathway for over-representation of `de_genes`.

    Args:
        de_genes:         gene symbols of interest (typically padj < 0.05 from
                          a differential expression test)
        background_genes: what to compare them against. Varying this, with
                          everything else held constant, is what this project
                          studies.
        gmt:              pathway definitions from load_gmt()
        min_term_size:    skip pathways with fewer than this many genes in the
                          background
        max_term_size:    skip pathways with more than this many

    Returns:
        One row per tested pathway, sorted by p-value, with columns:
            term_id, description
            term_size_in_background  n — pathway size within this background
            overlap                  k — query genes in this pathway
            n_de_in_background       K — query genes that could be scored
            background_size          N — background genes with annotations
            expected_overlap         nK/N
            fold_enrichment          k / expected
            pvalue                   one-sided hypergeometric, uncorrected
            padj                     Benjamini-Hochberg across tested pathways

        An empty frame with these columns if no pathway passes the size filter.
    """
    background, query = _restrict_to_annotated(de_genes, background_genes, gmt)

    term_ids, descriptions, overlaps, sizes = _pathway_sizes_within_background(
        gmt, background, query, min_term_size, max_term_size)

    if not term_ids:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    pvalues, padj, expected, fold_enrichment = _score(
        overlaps, sizes, len(background), len(query))

    results = pd.DataFrame({
        "term_id": term_ids,
        "description": descriptions,
        "term_size_in_background": np.asarray(sizes),
        "overlap": np.asarray(overlaps),
        "n_de_in_background": len(query),
        "background_size": len(background),
        "expected_overlap": expected,
        "fold_enrichment": fold_enrichment,
        "pvalue": pvalues,
        "padj": padj,
    })
    return results.sort_values("pvalue", kind="stable").reset_index(drop=True)
