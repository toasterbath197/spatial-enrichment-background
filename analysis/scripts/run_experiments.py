"""
Every enrichment experiment in the paper, in one place.

    python3 run_experiments.py --help              list the experiments
    python3 run_experiments.py baseline            the main panel-vs-genome comparison
    python3 run_experiments.py all                 run everything in order

Each experiment answers one question:

    baseline             Does background choice change which pathways are significant?
    simulation           Is the effect arithmetic rather than biological?
    subsampling          Does the effect grow as the panel shrinks?
    negative-control     Is the excess significance false positives?
    detected-background  Does the field's recommended intermediate fix help here?
    regression           Does panel size predict inflation across studies?
    sensitivity          How much do two unavoidable design choices cost us?

All of them read results/DE_tables/master_DE_table.csv and write to
results/enrichment_tables/. Output filenames are unchanged from when these were
seven separate scripts, so figures and verification still find them.

Pathway database is Reactome by default; pass --db go_bp to reproduce the
superseded first-pass results (see enrichment.py for why they are superseded).
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import hypergeom

from de_pipeline import detection_rates, enrichment as en
from de_pipeline.paths import ENRICHMENT_DIR, MASTER_DE_TABLE

DE_TABLE = MASTER_DE_TABLE
OUT_DIR = ENRICHMENT_DIR

PADJ = 0.05          # significance threshold, used throughout
MIN_QUERY_GENES = 3  # below this a gene list is too small to test meaningfully
SEED = 0             # fixed so every random draw is reproducible


# ===========================================================================
# Shared helpers
# ===========================================================================

def suffix(db):
    """Output files are db-suffixed so the databases never overwrite each
    other. go_bp came first and its files are unsuffixed, for continuity."""
    return "" if db == "go_bp" else f"_{db}"


def load_gene_lists():
    """The 78 self-computed gene lists, one group per (dataset x comparison x
    cell type). Author-published lists are excluded so that DE method is held
    constant across everything we compare."""
    de_table = pd.read_csv(DE_TABLE)
    return de_table[de_table["source"] == "self_computed"].groupby("list_id")


def pathway_data(species_field, db):
    """Pathway definitions and the genome-wide background for one dataset."""
    species = en.species_key(species_field)
    return en.load_gmt(species, db), en.gmt_universe(species, db), species


def count_significant(results):
    return int((results["padj"] < PADJ).sum()) if len(results) else 0


def best_pvalue(results):
    """Smallest uncorrected p-value across all tested pathways. This is the
    continuous companion to the significance count, which floors at zero and
    so cannot show graded differences."""
    return float(results["pvalue"].min()) if len(results) else float("nan")


def compare_two_backgrounds(query_genes, panel_genes, universe, gmt, list_id):
    """Run the same gene list against panel and genome-wide backgrounds.
    This single comparison is what the whole paper is built on."""
    scored = {}
    for name, background in (("panel_specific", panel_genes), ("genome_wide", universe)):
        results = en.hypergeometric_enrichment(query_genes, background, gmt)
        results.insert(0, "list_id", list_id)
        results.insert(1, "background_type", name)
        scored[name] = results
    return scored["panel_specific"], scored["genome_wide"]


def write(frame, stem, what):
    """Save one output table. `stem` is the complete filename without .csv, so
    the exact name is visible where it is written rather than assembled here."""
    path = OUT_DIR / f"{stem}.csv"
    frame.to_csv(path, index=False)
    print(f"  wrote {len(frame):,} {what} -> {path.name}")
    return path


# ===========================================================================
# baseline — the main result (Figure 1)
# ===========================================================================

def experiment_baseline(db):
    """Same gene list, two backgrounds, everything else held constant.

    This is the paper's central comparison. If background choice were
    immaterial, the two arms would agree.
    """
    print("BASELINE — panel-specific vs genome-wide background\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    term_rows, summaries = [], []
    for list_id, rows in load_gene_lists():
        gmt, universe, species = pathway_data(rows["species"].iloc[0], db)
        panel = rows["gene"].unique()
        query = rows.loc[rows["padj"] < PADJ, "gene"].unique()

        record = dict(
            paper_id=rows["paper_id"].iloc[0], list_id=list_id,
            species=rows["species"].iloc[0], species_key=species,
            platform=rows["platform"].iloc[0], cell_type=rows["cell_type"].iloc[0],
            comparison_type=rows["comparison_type"].iloc[0],
            panel_size=len(panel), panel_size_in_gmt_universe=len(set(panel) & universe),
            n_de_genes=len(query), n_de_genes_in_panel_universe=len(set(query) & universe),
            genome_universe_size=len(universe), pathway_db=db,
        )

        if len(query) < MIN_QUERY_GENES:
            summaries.append({**record, "skipped": True,
                              "reason": f"<{MIN_QUERY_GENES} DE genes"})
            print(f"  {list_id:70s} skipped, too few DE genes")
            continue

        panel_res, genome_res = compare_two_backgrounds(query, panel, universe, gmt, list_id)
        term_rows += [panel_res, genome_res]
        summaries.append({
            **record, "skipped": False, "reason": "",
            "n_terms_tested_panel_bg": len(panel_res),
            "n_terms_tested_genome_bg": len(genome_res),
            "n_sig_terms_panel_bg": count_significant(panel_res),
            "n_sig_terms_genome_bg": count_significant(genome_res),
        })
        print(f"  {list_id:70s} panel={count_significant(panel_res):4d}  "
              f"genome={count_significant(genome_res):4d}")

    summary = pd.DataFrame(summaries)
    write(pd.concat(term_rows, ignore_index=True), f"step0_baseline_enrichment{suffix(db)}", "pathway rows")
    write(summary, f"step0_baseline_summary{suffix(db)}", "gene lists")

    scored = summary[~summary["skipped"]]
    print(f"\n  {len(scored)} of {len(summary)} lists scored")
    print(f"  mean significant pathways: panel {scored['n_sig_terms_panel_bg'].mean():.3f}, "
          f"genome {scored['n_sig_terms_genome_bg'].mean():.2f}")
    print(f"  genome background gives more on "
          f"{int((scored['n_sig_terms_genome_bg'] > scored['n_sig_terms_panel_bg']).sum())}"
          f"/{len(scored)} lists")
    return summary


# ===========================================================================
# negative-control — is the excess just false positives? (Figure 4)
# ===========================================================================

def experiment_negative_control(db):
    """Same pipeline, but the query is genes the DE test called NON-significant.

    These are real measured genes known to carry no signal for that contrast,
    so any pathway reaching significance is a false positive by construction.
    """
    print("NEGATIVE CONTROL — querying genes known to carry no signal\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    term_rows, summaries = [], []
    for list_id, rows in load_gene_lists():
        gmt, universe, species = pathway_data(rows["species"].iloc[0], db)
        panel = rows["gene"].unique()
        junk = rows.loc[rows["padj"] >= PADJ, "gene"].unique()   # the complement of the DE list

        record = dict(
            paper_id=rows["paper_id"].iloc[0], list_id=list_id, species_key=species,
            panel_size=len(panel), n_junk_genes=len(junk),
            genome_universe_size=len(universe),
        )

        if len(junk) < MIN_QUERY_GENES:
            summaries.append({**record, "skipped": True,
                              "reason": f"<{MIN_QUERY_GENES} junk genes"})
            continue

        panel_res, genome_res = compare_two_backgrounds(junk, panel, universe, gmt, list_id)
        term_rows += [panel_res, genome_res]
        summaries.append({
            **record, "skipped": False, "reason": "",
            "n_terms_tested_panel_bg": len(panel_res),
            "n_terms_tested_genome_bg": len(genome_res),
            "n_false_hits_panel_bg": count_significant(panel_res),
            "n_false_hits_genome_bg": count_significant(genome_res),
        })
        print(f"  {list_id:70s} panel={count_significant(panel_res):4d}  "
              f"genome={count_significant(genome_res):4d}")

    summary = pd.DataFrame(summaries)
    write(pd.concat(term_rows, ignore_index=True), f"fix2_negative_control_enrichment{suffix(db)}", "pathway rows")
    write(summary, f"fix2_negative_control_summary{suffix(db)}", "gene lists")

    scored = summary[~summary["skipped"]]
    print(f"\n  mean false positives: panel {scored['n_false_hits_panel_bg'].mean():.3f}, "
          f"genome {scored['n_false_hits_genome_bg'].mean():.3f}")
    print(f"  lists with at least one false positive: "
          f"panel {(scored['n_false_hits_panel_bg'] > 0).mean() * 100:.1f}%, "
          f"genome {(scored['n_false_hits_genome_bg'] > 0).mean() * 100:.1f}%")
    return summary


# ===========================================================================
# simulation — the mechanism, with no real data at all (Figure 2)
# ===========================================================================

PATHWAY_SIZE, QUERY_SIZE, TRUE_OVERLAP = 50, 40, 15
BACKGROUND_SIZES = [100, 150, 250, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]


def experiment_simulation(db=None):
    """A made-up pathway, a made-up gene list, a fixed overlap - and only the
    background size varies.

    No biological data is involved, so nothing about any dataset can influence
    the result. Assertions enforce that the three quantities that should stay
    fixed actually do; if the engine mishandled the background, they would trip.
    """
    print("SIMULATION — only background size varies, biology held fixed\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pathway = frozenset(f"PATH_{i:06d}" for i in range(PATHWAY_SIZE))
    query = frozenset(sorted(pathway)[:TRUE_OVERLAP]) | frozenset(
        f"DEONLY_{i:06d}" for i in range(QUERY_SIZE - TRUE_OVERLAP))
    assert len(query) == QUERY_SIZE

    rows = []
    print(f"  {'N':>8s}  {'size':>5s}  {'overlap':>7s}  {'fold':>9s}  {'p-value':>12s}")
    for n_background in BACKGROUND_SIZES:
        core = pathway | query
        filler = {f"FILLER_{i:06d}" for i in range(n_background - len(core))}
        background = core | filler

        # A real database spans thousands of genes, so a query gene outside the
        # pathway under test still counts toward K because some other pathway
        # annotates it. This second term reproduces that, so the engine's
        # universe equals the whole background at every N. Only the real
        # pathway's row is kept.
        gmt = {
            "SYNTHETIC_PATHWAY": ("the pathway under test", pathway),
            "SYNTHETIC_FILLER": ("bookkeeping so the universe spans the background",
                                 frozenset(background - pathway)),
        }
        result = en.hypergeometric_enrichment(
            query, background, gmt, min_term_size=1, max_term_size=10**9)
        result = result[result["term_id"] == "SYNTHETIC_PATHWAY"]
        assert len(result) == 1
        row = result.iloc[0]

        # these three must not move; if they do, the engine has a bug
        assert int(row["term_size_in_background"]) == PATHWAY_SIZE
        assert int(row["overlap"]) == TRUE_OVERLAP
        assert int(row["n_de_in_background"]) == QUERY_SIZE

        rows.append({
            "background_size_N": n_background,
            "term_size_in_background": int(row["term_size_in_background"]),
            "overlap": int(row["overlap"]),
            "n_de_in_background": int(row["n_de_in_background"]),
            "expected_overlap": float(row["expected_overlap"]),
            "fold_enrichment": float(row["fold_enrichment"]),
            "pvalue": float(row["pvalue"]),
            "padj": float(row["padj"]),
        })
        print(f"  {n_background:8d}  {PATHWAY_SIZE:5d}  {TRUE_OVERLAP:7d}  "
              f"{row['fold_enrichment']:9.2f}  {row['pvalue']:12.3e}")

    table = pd.DataFrame(rows)
    path = OUT_DIR / "fix8_synthetic_simulation.csv"   # no db suffix: no database involved
    table.to_csv(path, index=False)
    print(f"\n  wrote {len(table)} rows -> {path.name}")
    print(f"  p-value fell from {table['pvalue'].iloc[0]:.3f} to {table['pvalue'].iloc[-1]:.2e} "
          f"with the biology unchanged")
    return table


# ===========================================================================
# subsampling — does the effect scale with panel size? (Figure 3)
# ===========================================================================

SUBSAMPLING_DATASETS = {
    "paper03": ("03_CosMx_AmyloidPlaqueNiche__proximal_vs_distal__all",
                [900, 800, 600, 400, 200, 100, 50, 25], "fix1_paper03_subsampling"),
    "paper07b": ("07b_MERFISH_PU1_LymphoidMicroglia__FADPU_vs_FADTV__all",
                 [350, 300, 200, 100, 50, 25], "fix1_paper07b_subsampling"),
    "paper08": ("08_MERFISH_TcellMyelin__Tcell_neighbor_vs_nonneighbor__all",
                [450, 400, 300, 200, 100, 50, 25], "fix9_paper08_subsampling"),
}


def experiment_subsampling(db, dataset="paper03", n_draws=100):
    """Shrink one dataset's panel at random and watch the gap widen.

    Held fixed: the DE gene list (taken from the full-panel analysis and never
    recomputed) and the genome-wide arm (which does not depend on panel size).
    Varied: panel size, and nothing else.
    """
    list_id, size_ladder, out_name = SUBSAMPLING_DATASETS[dataset]
    print(f"SUBSAMPLING — {dataset}, {n_draws} draws per panel size\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    de_table = pd.read_csv(DE_TABLE)
    rows = de_table[(de_table["list_id"] == list_id) & (de_table["source"] == "self_computed")]
    assert len(rows), f"gene list not found: {list_id}"

    full_panel = rows["gene"].unique()
    frozen_query = rows.loc[rows["padj"] < PADJ, "gene"].unique()
    gmt, universe, _species = pathway_data(rows["species"].iloc[0], db)

    genome_result = en.hypergeometric_enrichment(frozen_query, universe, gmt)
    genome_significant = count_significant(genome_result)
    genome_best_p = best_pvalue(genome_result)
    print(f"  full panel {len(full_panel)} genes, {len(frozen_query)} DE genes")
    print(f"  genome-wide reference (fixed at every size): "
          f"{genome_significant} pathways, best p = {genome_best_p:.3e}\n")

    rng = np.random.default_rng(SEED)
    draws = []
    for size in size_ladder:
        if size > len(full_panel):
            continue
        for draw in range(n_draws):
            shrunk = rng.choice(full_panel, size=size, replace=False)
            result = en.hypergeometric_enrichment(frozen_query, shrunk, gmt)
            panel_best_p = best_pvalue(result)
            draws.append({
                "panel_size": size, "draw_idx": draw,
                "n_de_genes_retained": len(set(frozen_query) & set(shrunk)),
                "n_sig_terms_panel_bg": count_significant(result),
                "n_sig_terms_genome_bg": genome_significant,
                "inflation_gap": genome_significant - count_significant(result),
                "min_pvalue_panel_bg": panel_best_p,
                "min_pvalue_genome_bg": genome_best_p,
                "neglog10_pvalue_gap": -np.log10(genome_best_p) + np.log10(panel_best_p)
                if panel_best_p > 0 and genome_best_p > 0 else np.nan,
            })
        print(f"  panel size {size:4d}: done")

    draws = pd.DataFrame(draws)
    summary = draws.groupby("panel_size").agg(
        n_draws=("draw_idx", "count"),
        mean_n_de_retained=("n_de_genes_retained", "mean"),
        mean_n_sig_panel_bg=("n_sig_terms_panel_bg", "mean"),
        std_n_sig_panel_bg=("n_sig_terms_panel_bg", "std"),
        mean_inflation_gap=("inflation_gap", "mean"),
        std_inflation_gap=("inflation_gap", "std"),
        mean_min_pvalue_panel_bg=("min_pvalue_panel_bg", "mean"),
        mean_neglog10_pvalue_gap=("neglog10_pvalue_gap", "mean"),
        std_neglog10_pvalue_gap=("neglog10_pvalue_gap", "std"),
    ).reset_index().sort_values("panel_size", ascending=False)
    summary["n_sig_genome_bg_fixed"] = genome_significant
    summary["min_pvalue_genome_bg_fixed"] = genome_best_p

    write(draws, f"{out_name}{suffix(db)}_draws", "individual draws")
    write(summary, f"{out_name}{suffix(db)}_summary", "panel sizes")

    largest, smallest = summary.iloc[0], summary.iloc[-1]
    print(f"\n  gap at {int(largest['panel_size'])} genes: "
          f"{largest['mean_neglog10_pvalue_gap']:.2f}")
    print(f"  gap at {int(smallest['panel_size'])} genes: "
          f"{smallest['mean_neglog10_pvalue_gap']:.2f}")
    return summary


# ===========================================================================
# detected-background — does the recommended fix help? (Figure 5)
# ===========================================================================

def experiment_detected_background(db, only_dataset=None):
    """Add the intermediate background current guidance recommends: genes
    actually detected in the data, rather than every gene on the panel.

    For whole-transcriptome data this sits meaningfully between panel and
    genome. The question is whether it does so for targeted panels.
    """
    print("DETECTED BACKGROUND — testing the field's recommended middle option\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    de_table = pd.read_csv(DE_TABLE)
    de_table = de_table[de_table["source"] == "self_computed"]

    rows = []
    for paper_id, get_detection_rates in detection_rates.READERS.items():
        if only_dataset and paper_id != only_dataset:
            continue
        paper_rows = de_table[de_table["paper_id"] == paper_id]
        assert len(paper_rows), f"dataset not found: {paper_id}"
        gmt, universe, _species = pathway_data(paper_rows["species"].iloc[0], db)

        print(f"  {paper_id}")
        rates = get_detection_rates()          # computed once per dataset, not per list

        for list_id, list_rows in paper_rows.groupby("list_id"):
            panel = list_rows["gene"].unique()
            query = list_rows.loc[list_rows["padj"] < PADJ, "gene"].unique()
            if len(query) < MIN_QUERY_GENES:
                rows.append({"paper_id": paper_id, "list_id": list_id,
                             "background_type": None, "background_size": None,
                             "n_terms_tested": None, "n_sig_terms_padj05": None,
                             "min_pvalue": None, "skipped": True,
                             "reason": f"<{MIN_QUERY_GENES} DE genes"})
                continue

            detected = np.array([g for g in panel
                                 if rates.get(g, 0.0) >= detection_rates.DEFAULT_THRESHOLD])
            for name, background in (("panel_specific", panel),
                                     ("detected_genes", detected),
                                     ("genome_wide", np.array(list(universe)))):
                result = en.hypergeometric_enrichment(query, background, gmt)
                rows.append({
                    "paper_id": paper_id, "list_id": list_id, "background_type": name,
                    "background_size": len(background), "n_terms_tested": len(result),
                    "n_sig_terms_padj05": count_significant(result),
                    "min_pvalue": best_pvalue(result), "skipped": False, "reason": "",
                })
            print(f"    {list_id[:64]:64s} detected {len(detected)}/{len(panel)}")

    table = pd.DataFrame(rows)
    name = "fix3_detected_background" + (f"_{only_dataset}" if only_dataset else "")
    write(table, f"{name}{suffix(db)}", "rows")
    return table


# ===========================================================================
# regression — does panel size predict inflation across studies? (Figure 6)
# ===========================================================================

def experiment_regression(db):
    """Regress the inflation gap on panel size across all scored lists, before
    and after adjusting for the other things that differ between studies.

    This is exploratory. The controlled subsampling experiment carries the
    argument about panel size; this only shows what a naive across-study
    comparison would tell you.
    """
    print("REGRESSION — panel size vs inflation, across studies\n")
    summary = pd.read_csv(OUT_DIR / f"step0_baseline_summary{suffix(db)}.csv")
    summary = summary[~summary["skipped"]].copy()

    # recover the continuous metric per list from the term-level baseline output
    terms = pd.read_csv(OUT_DIR / f"step0_baseline_enrichment{suffix(db)}.csv")
    best = terms.groupby(["list_id", "background_type"])["pvalue"].min().unstack()
    best.columns = [f"min_pvalue_{c}" for c in best.columns]
    summary = summary.merge(best, left_on="list_id", right_index=True, how="left")

    summary["count_gap"] = summary["n_sig_terms_genome_bg"] - summary["n_sig_terms_panel_bg"]
    with np.errstate(divide="ignore"):
        summary["neglog10_gap"] = (
            -np.log10(summary["min_pvalue_genome_wide"].replace(0, np.nan))
            + np.log10(summary["min_pvalue_panel_specific"].replace(0, np.nan)))
    summary["species_binary"] = summary["species"].str.contains("Human", case=False).astype(int)

    usable = summary.dropna(subset=["neglog10_gap", "panel_size", "n_de_genes"])
    print(f"  {len(usable)} of {len(summary)} lists usable\n")

    models = {}
    for label, predictors in (
        ("(a) unadjusted: gap ~ panel_size", ["panel_size"]),
        ("(b) adjusted: gap ~ panel_size + n_de_genes + species",
         ["panel_size", "n_de_genes", "species_binary"]),
    ):
        fitted = sm.OLS(usable["neglog10_gap"], sm.add_constant(usable[predictors])).fit()
        models[label] = fitted
        print(f"  {label}")
        print(f"    panel_size coefficient {fitted.params['panel_size']:+.5f}  "
              f"p = {fitted.pvalues['panel_size']:.4g}  R2 = {fitted.rsquared:.3f}")

    coefficients = pd.DataFrame([
        {"model": label, "covariate": name,
         "coefficient": fitted.params[name], "pvalue": fitted.pvalues[name],
         "r_squared": fitted.rsquared}
        for label, fitted in models.items() for name in fitted.params.index
    ])
    write(coefficients, f"fix6b_adjustment_regression{suffix(db)}", "coefficients")
    write(summary, f"fix6b_adjustment_data{suffix(db)}", "lists")
    return coefficients


# ===========================================================================
# sensitivity — bounding two design choices we cannot remove
# ===========================================================================

PADDED_BACKGROUND_SIZES = [15000, 20000, 25000]


def experiment_sensitivity(db):
    """Two questions the design cannot avoid, answered by measurement.

    A. Our genome-wide arm is an annotation universe (~9,200 genes), not the
       whole genome. What would the results look like against a fuller one?
       Answered by holding the counts fixed and raising only N — computed
       directly, because the engine correctly discards unannotated genes and
       so cannot be asked this question by passing a padded background.

    B. Dataset 08 keeps its source paper's Bonferroni correction rather than
       our Benjamini-Hochberg default. Does that change the conclusion?
    """
    print("SENSITIVITY — bounding two unavoidable design choices\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- A. what a fuller background would show ---------------------------
    rows = []
    for list_id, list_rows in load_gene_lists():
        query = list_rows.loc[list_rows["padj"] < PADJ, "gene"].unique()
        if len(query) < MIN_QUERY_GENES:
            continue
        panel = list_rows["gene"].unique()
        gmt, universe, species = pathway_data(list_rows["species"].iloc[0], db)

        panel_best_p = best_pvalue(en.hypergeometric_enrichment(query, panel, gmt))
        genome_result = en.hypergeometric_enrichment(query, universe, gmt)
        if not len(genome_result):
            continue

        real_n = len(universe)
        for name, n_background in ([("annotation_universe", real_n)]
                                   + [(f"padded_{n}", n) for n in PADDED_BACKGROUND_SIZES
                                      if n > real_n]):
            if name == "annotation_universe":
                pvalues = genome_result["pvalue"].values
            else:
                # same counts, larger N — this is the whole point
                pvalues = hypergeom.sf(
                    genome_result["overlap"].values - 1, n_background,
                    genome_result["term_size_in_background"].values,
                    int(genome_result["n_de_in_background"].iloc[0]))
            from statsmodels.stats.multitest import multipletests
            padj = multipletests(pvalues, method="fdr_bh")[1]
            gap = -np.log10(np.min(pvalues)) + np.log10(panel_best_p) if panel_best_p > 0 else np.nan
            rows.append({
                "list_id": list_id, "paper_id": list_rows["paper_id"].iloc[0],
                "species": species, "background_variant": name,
                "background_size": n_background, "panel_size": len(panel),
                "n_sig_terms": int((padj < PADJ).sum()),
                "min_pvalue": float(np.min(pvalues)),
                "min_pvalue_panel_bg": panel_best_p,
                "neglog10_gap_vs_panel": gap,
            })

    background_table = pd.DataFrame(rows)
    write(background_table, f"fix10_sensitivity_background{suffix(db)}", "rows")
    by_variant = background_table.groupby("background_variant")["n_sig_terms"].mean()
    print("\n  mean significant pathways by background size:")
    for name in ["annotation_universe"] + [f"padded_{n}" for n in PADDED_BACKGROUND_SIZES]:
        if name in by_variant:
            print(f"    {name:22s} {by_variant[name]:7.1f}")

    # --- B. does dataset 08's correction method matter? -------------------
    from statsmodels.stats.multitest import multipletests
    de_table = pd.read_csv(DE_TABLE)
    p08 = de_table[(de_table["paper_id"] == "08_MERFISH_TcellMyelin")
                   & (de_table["source"] == "self_computed")].copy()
    p08["padj_bh"] = multipletests(p08["pvalue"].values, method="fdr_bh")[1]
    gmt, universe, _species = pathway_data(p08["species"].iloc[0], db)
    panel = p08["gene"].unique()

    rows = []
    for label, column in (("bonferroni (as published, primary)", "padj"),
                          ("benjamini_hochberg (project standard)", "padj_bh")):
        query = p08.loc[p08[column] < PADJ, "gene"].unique()
        for name, background in (("panel_specific", panel),
                                 ("genome_wide", np.array(list(universe)))):
            result = en.hypergeometric_enrichment(query, background, gmt)
            rows.append({"correction": label, "n_de_genes": len(query),
                         "background_type": name, "background_size": len(background),
                         "n_sig_terms": count_significant(result),
                         "min_pvalue": best_pvalue(result)})

    correction_table = pd.DataFrame(rows)
    write(correction_table, f"fix10_sensitivity_correction{suffix(db)}", "rows")
    print("\n  dataset 08 under both corrections:")
    print(correction_table.to_string(index=False).replace("\n", "\n  "))
    return background_table, correction_table


# ===========================================================================

EXPERIMENTS = {
    "baseline": experiment_baseline,
    "simulation": experiment_simulation,
    "subsampling": experiment_subsampling,
    "negative-control": experiment_negative_control,
    "detected-background": experiment_detected_background,
    "regression": experiment_regression,
    "sensitivity": experiment_sensitivity,
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=list(EXPERIMENTS) + ["all"],
                        help="which experiment to run")
    parser.add_argument("--db", default=en.DEFAULT_DB,
                        choices=["reactome", "go_bp", "wikipathways"],
                        help="pathway database (default: reactome)")
    parser.add_argument("--dataset", default="paper03",
                        help="for subsampling: paper03, paper07b or paper08")
    parser.add_argument("--draws", type=int, default=100,
                        help="for subsampling: random draws per panel size")
    args = parser.parse_args()

    if args.experiment == "all":
        # regression reads baseline's output, so order matters
        experiment_baseline(args.db)
        experiment_simulation()
        for dataset in SUBSAMPLING_DATASETS:
            experiment_subsampling(args.db, dataset, args.draws)
        experiment_negative_control(args.db)
        experiment_detected_background(args.db)
        experiment_regression(args.db)
        experiment_sensitivity(args.db)
        return

    if args.experiment == "subsampling":
        experiment_subsampling(args.db, args.dataset, args.draws)
    elif args.experiment == "simulation":
        experiment_simulation()
    else:
        EXPERIMENTS[args.experiment](args.db)


if __name__ == "__main__":
    main()
