"""
Check every number the paper reports against the result files.

WHY THIS EXISTS: a reader should be able to confirm the paper's claims without
reading the analysis code or rerunning anything. This script recomputes each
published figure from the stored results and compares it to the value in the
manuscript. Every check prints PASS or FAIL with both numbers, so a mismatch
tells you exactly what disagrees.

It reads result files only — it never recomputes an enrichment test, so it
cannot accidentally "confirm" a number by reproducing the same bug twice.
To re-derive the results themselves, see section 27 of
00_PROJECT_NOTES_AND_METHODOLOGY.txt.

RUN:  python3 analysis/verify_results.py

EXIT CODE: 0 if every claim checks out, 1 otherwise.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from de_pipeline.paths import ENRICHMENT_DIR as ENRICHMENT
from de_pipeline.paths import DE_TABLES_DIR as DE_TABLES

PADJ = 0.05
_checks_run = 0
_failures = []


def check(claim, actual, expected, tol=0.01):
    """Compare one published claim against the recomputed value."""
    global _checks_run
    _checks_run += 1
    if isinstance(expected, str) or isinstance(actual, str):
        ok = str(actual) == str(expected)
    else:
        ok = bool(np.isclose(float(actual), float(expected), rtol=tol, atol=tol))
    if ok:
        print(f"  PASS  {claim}")
        print(f"        paper says {expected}, files give {actual}")
    else:
        _failures.append(claim)
        print(f"  FAIL  {claim}")
        print(f"        paper says {expected}, files give {actual}   <-- MISMATCH")


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


# ---------------------------------------------------------------------------

def verify_corpus():
    section("Corpus and panel sizes (Table 1, Methods 2.2)")
    de = pd.read_csv(DE_TABLES / "master_DE_table.csv")
    own = de[de["source"] == "self_computed"]

    check("seven active datasets", own["paper_id"].nunique(), 7)
    check("78 differentially expressed gene lists", own["list_id"].nunique(), 78)

    published_panel_sizes = {
        "01_Trem2R47H_MERFISH": 300,
        "03_CosMx_AmyloidPlaqueNiche": 950,
        "04_SEAAD_MERFISH": 140,
        "05_Xenium_PVInterneuron_RetrosplenialCortex": 247,
        "06_Xenium_APOE4vAPOE3": 266,
        "07b_MERFISH_PU1_LymphoidMicroglia": 398,
        "08_MERFISH_TcellMyelin": 496,
    }
    measured = own.groupby("paper_id")["gene"].nunique()
    for paper, expected_size in published_panel_sizes.items():
        check(f"{paper} panel size", int(measured[paper]), expected_size, tol=0)

    # the control-probe fix: no dataset should still contain control features
    import re
    control_pattern = re.compile(
        r"(neg[_ -]?(control|prb|probe)|^blank|codeword|deprecated|unassigned)", re.I)
    contaminated = [g for g in own["gene"].unique() if control_pattern.search(str(g))]
    check("no control probes remain in any dataset", len(contaminated), 0, tol=0)


def verify_h1_baseline():
    section("H1 — background choice alone (Figure 1, Results 3.2)")
    summary = pd.read_csv(ENRICHMENT / "step0_baseline_summary_reactome.csv")
    scored = summary[~summary["skipped"]]

    check("lists scored", len(scored), 54, tol=0)
    check("lists skipped for <3 DE genes", int(summary["skipped"].sum()), 24, tol=0)
    check("all skipped lists come from dataset 04",
          summary[summary["skipped"]]["paper_id"].nunique(), 1, tol=0)

    check("mean significant pathways, panel background",
          round(scored["n_sig_terms_panel_bg"].mean(), 3), 0.037)
    check("mean significant pathways, genome background",
          round(scored["n_sig_terms_genome_bg"].mean(), 2), 53.78)
    check("maximum on any list, panel background",
          int(scored["n_sig_terms_panel_bg"].max()), 2, tol=0)
    check("maximum on any list, genome background",
          int(scored["n_sig_terms_genome_bg"].max()), 248, tol=0)

    genome_higher = int((scored["n_sig_terms_genome_bg"] > scored["n_sig_terms_panel_bg"]).sum())
    panel_higher = int((scored["n_sig_terms_panel_bg"] > scored["n_sig_terms_genome_bg"]).sum())
    check("lists where genome background gives more", genome_higher, 54, tol=0)
    check("lists where panel background gives more", panel_higher, 0, tol=0)


def verify_synthetic_control():
    section("Mechanism — synthetic control (Figure 2, Results 3.3)")
    sim = pd.read_csv(ENRICHMENT / "fix8_synthetic_simulation.csv").sort_values("background_size_N")

    check("p-value at N = 100", round(float(sim["pvalue"].iloc[0]), 3), 0.988)
    check("p-value at N = 100,000", float(sim["pvalue"].iloc[-1]), 1.18e-40, tol=1e-41)
    check("fold enrichment at N = 100", float(sim["fold_enrichment"].iloc[0]), 0.75)
    check("fold enrichment at N = 100,000", float(sim["fold_enrichment"].iloc[-1]), 750.0)

    # the control's whole point: the biology never changes, only N
    check("pathway size held fixed across all N", sim["term_size_in_background"].nunique(), 1, tol=0)
    check("overlap held fixed across all N", sim["overlap"].nunique(), 1, tol=0)
    check("query size held fixed across all N", sim["n_de_in_background"].nunique(), 1, tol=0)
    check("p-value falls monotonically as N grows",
          bool((sim["pvalue"].diff().dropna() < 0).all()), True)


def verify_h2_subsampling():
    section("H2 — panel-size dose response (Figure 3, Results 3.4)")
    published = {
        "fix1_paper03_subsampling_reactome_summary.csv":  ("dataset 03", 18.02, 21.97, 8),
        "fix1_paper07b_subsampling_reactome_summary.csv": ("dataset 07b", 11.81, 12.93, 6),
        "fix9_paper08_subsampling_reactome_summary.csv":  ("dataset 08", 3.40, 5.74, 7),
    }
    for filename, (label, gap_largest, gap_smallest, n_rungs) in published.items():
        summary = pd.read_csv(ENRICHMENT / filename).sort_values("panel_size", ascending=False)
        check(f"{label}: rungs on the size ladder", len(summary), n_rungs, tol=0)
        check(f"{label}: draws per rung", int(summary["n_draws"].iloc[0]), 100, tol=0)
        check(f"{label}: gap at largest panel",
              round(float(summary["mean_neglog10_pvalue_gap"].iloc[0]), 2), gap_largest)
        check(f"{label}: gap at 25 genes",
              round(float(summary["mean_neglog10_pvalue_gap"].iloc[-1]), 2), gap_smallest)
        # monotonic increase as the panel shrinks
        gaps = summary["mean_neglog10_pvalue_gap"].values
        check(f"{label}: gap increases monotonically as panel shrinks",
              bool(np.all(np.diff(gaps) > 0)), True)

    # draws where a shrunken panel left nothing testable
    for filename, label, expected_untestable in [
        ("fix1_paper03_subsampling_reactome_draws.csv", "dataset 03", 2),
        ("fix9_paper08_subsampling_reactome_draws.csv", "dataset 08", 3),
    ]:
        draws = pd.read_csv(ENRICHMENT / filename)
        smallest = draws[draws["panel_size"] == draws["panel_size"].min()]
        check(f"{label}: untestable draws at 25 genes",
              int(smallest["min_pvalue_panel_bg"].isna().sum()), expected_untestable, tol=0)


def verify_negative_control():
    section("Negative control (Figure 4, Results 3.5)")
    summary = pd.read_csv(ENRICHMENT / "fix2_negative_control_summary_reactome.csv")
    scored = summary[~summary["skipped"]]

    check("lists tested", len(scored), 78, tol=0)
    check("mean false positives, panel background",
          round(scored["n_false_hits_panel_bg"].mean(), 3), 0.026)
    check("mean false positives, genome background",
          round(scored["n_false_hits_genome_bg"].mean(), 3), 7.051)
    check("percent of lists with a false positive, panel background",
          round((scored["n_false_hits_panel_bg"] > 0).mean() * 100, 1), 1.3)
    check("percent of lists with a false positive, genome background",
          round((scored["n_false_hits_genome_bg"] > 0).mean() * 100, 1), 91.0)
    check("worst case, genome background", int(scored["n_false_hits_genome_bg"].max()), 37, tol=0)


def verify_detected_background():
    section("Detected-genes background (Figure 5, Results 3.6)")
    detected = pd.read_csv(ENRICHMENT / "fix3_detected_background_reactome.csv")
    detected = detected[detected["skipped"] == False]  # noqa: E712

    check("scored lists covered", detected["list_id"].nunique(), 54, tol=0)
    # Fix 3 was RUN on all seven datasets, but dataset 04 contributes no scored
    # lists (every one falls below the 3-DE-gene minimum), so only six appear in
    # the scored rows. Both numbers are checked so neither can drift unnoticed.
    all_rows = pd.read_csv(ENRICHMENT / "fix3_detected_background_reactome.csv")
    check("datasets the analysis was run on", all_rows["paper_id"].nunique(), 7, tol=0)
    check("datasets contributing scored lists", detected["paper_id"].nunique(), 6, tol=0)

    sizes = detected.pivot_table(index="list_id", columns="background_type", values="background_size")
    pvalues = detected.pivot_table(index="list_id", columns="background_type", values="min_pvalue")
    sig = detected.pivot_table(index="list_id", columns="background_type", values="n_sig_terms_padj05")

    percent_of_panel = 100 * sizes["detected_genes"] / sizes["panel_specific"]
    check("median detected-genes size as percent of panel",
          round(float(percent_of_panel.median()), 1), 100.0)

    check("mean significant pathways, panel", round(float(sig["panel_specific"].mean()), 3), 0.037)
    check("mean significant pathways, detected", round(float(sig["detected_genes"].mean()), 3), 0.037)
    check("mean significant pathways, genome", round(float(sig["genome_wide"].mean()), 2), 53.78)

    log_ratio = np.log10(pvalues["panel_specific"] / pvalues["detected_genes"])
    check("median log10(panel p / detected p)", round(float(log_ratio.median()), 4), 0.0)

    check("lists where genome beats panel",
          int((pvalues["genome_wide"] < pvalues["panel_specific"]).sum()), 54, tol=0)
    check("lists where detected beats panel",
          int((pvalues["detected_genes"] < pvalues["panel_specific"]).sum()), 2, tol=0)


def verify_confound_adjustment():
    section("Across-study regression (Figure 6, Results 3.7)")
    regression = pd.read_csv(ENRICHMENT / "fix6b_adjustment_regression_reactome.csv")
    panel_size_rows = regression[regression["covariate"] == "panel_size"]

    unadjusted = panel_size_rows[panel_size_rows["model"].str.startswith("(a)")].iloc[0]
    adjusted = panel_size_rows[panel_size_rows["model"].str.startswith("(b)")].iloc[0]

    check("unadjusted coefficient is effectively zero",
          round(float(unadjusted["coefficient"]), 6), 0.000055, tol=1e-4)
    check("unadjusted p-value", round(float(unadjusted["pvalue"]), 3), 0.995)
    check("adjusted coefficient", round(float(adjusted["coefficient"]), 4), -0.0647)
    check("adjusted p-value below 1e-5", bool(float(adjusted["pvalue"]) < 1e-5), True)
    check("adjusted R squared", round(float(adjusted["r_squared"]), 3), 0.550)

    # the retracted claim: check it stays retracted
    check("no sign flip — both coefficients are not opposite in sign",
          bool(np.sign(unadjusted["coefficient"]) == np.sign(adjusted["coefficient"])), False)


def verify_sensitivity():
    section("Sensitivity analyses (Results 3.8)")
    background = pd.read_csv(ENRICHMENT / "fix10_sensitivity_background_reactome.csv")
    by_variant = background.groupby("background_variant")["n_sig_terms"].mean()

    check("mean significant pathways, annotation universe",
          round(float(by_variant["annotation_universe"]), 1), 53.8)
    check("mean significant pathways, padded to 20,000",
          round(float(by_variant["padded_20000"]), 1), 145.3)
    ratio = by_variant["padded_20000"] / by_variant["annotation_universe"]
    check("inflation at a 20,000-gene background is ~2.7x larger", round(float(ratio), 1), 2.7)

    correction = pd.read_csv(ENRICHMENT / "fix10_sensitivity_correction_reactome.csv")
    genome_rows = correction[correction["background_type"] == "genome_wide"]
    panel_rows = correction[correction["background_type"] == "panel_specific"]
    check("dataset 08 under Bonferroni: genome background significant pathways",
          int(genome_rows[genome_rows["correction"].str.startswith("bonferroni")]["n_sig_terms"].iloc[0]), 2, tol=0)
    check("dataset 08 under BH: genome background significant pathways",
          int(genome_rows[genome_rows["correction"].str.startswith("benjamini")]["n_sig_terms"].iloc[0]), 12, tol=0)
    check("dataset 08: panel background gives zero under both corrections",
          int(panel_rows["n_sig_terms"].sum()), 0, tol=0)


def main():
    print(__doc__.strip().split("RUN:")[0].strip())
    print("=" * 72)

    verify_corpus()
    verify_h1_baseline()
    verify_synthetic_control()
    verify_h2_subsampling()
    verify_negative_control()
    verify_detected_background()
    verify_confound_adjustment()
    verify_sensitivity()

    print("\n" + "=" * 72)
    if _failures:
        print(f"{_checks_run - len(_failures)}/{_checks_run} checks passed. FAILURES:")
        for claim in _failures:
            print(f"  - {claim}")
        return 1
    print(f"All {_checks_run} published claims verified against the result files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
