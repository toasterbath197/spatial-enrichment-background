"""
Builds every publication figure and formatted table for the write-up, from the
already-computed result CSVs in analysis/results/. Reads only - never recomputes
any statistic, so a figure can never silently disagree with the analysis that
produced it.

Run:  python3 build_figures.py
Writes: figures/*.png (300 dpi) and tables/*.csv in this same folder.
Captions for every panel: ../00_PROJECT_NOTES_AND_METHODOLOGY.txt, section 31.

Every number plotted here was verified against its source CSV on 2026-08-10
(see section 32 of ../00_PROJECT_NOTES_AND_METHODOLOGY.txt for the audit).
"""
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = pathlib.Path(__file__).resolve().parent
_PROJECT = _HERE.parent
ENR = _PROJECT / "analysis" / "results" / "enrichment_tables"
RES = _PROJECT / "analysis" / "results"
FIGDIR = _HERE / "figures"
TABDIR = _HERE / "tables"
FIGDIR.mkdir(parents=True, exist_ok=True)
TABDIR.mkdir(parents=True, exist_ok=True)

# consistent styling across every figure
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})
C_PANEL = "#2c6fb5"     # panel-specific background
C_GENOME = "#c1442e"    # genome-wide background
C_DETECT = "#5a9367"    # detected-genes background
C_NEUTRAL = "#666666"


def fig1_baseline():
    """H1: panel vs genome-wide background across all scored DE lists."""
    s = pd.read_csv(ENR / "step0_baseline_summary_reactome.csv")
    sc = s[~s["skipped"]].copy().sort_values("n_sig_terms_genome_bg", ascending=False)
    sc = sc.reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [2.1, 1]})

    ax = axes[0]
    x = np.arange(len(sc))
    ax.bar(x, sc["n_sig_terms_genome_bg"], color=C_GENOME, label="Genome-wide background", width=0.9)
    ax.bar(x, sc["n_sig_terms_panel_bg"], color=C_PANEL, label="Panel-specific background", width=0.9)
    ax.set_xlabel(f"Individual DE gene lists (n = {len(sc)}, sorted by genome-wide result)")
    ax.set_ylabel("Significant pathways (padj < 0.05)")
    ax.set_title("A. Same gene lists, two backgrounds", loc="left", fontweight="bold")
    ax.legend(loc="upper right")
    ax.annotate("panel-specific background = 0 significant terms\nfor every one of these lists (bars have zero height)",
                xy=(len(sc) * 0.62, 0.5), xytext=(len(sc) * 0.30, 170),
                fontsize=8, color=C_PANEL, ha="left",
                arrowprops=dict(arrowstyle="->", color=C_PANEL, lw=1.1,
                                connectionstyle="arc3,rad=-0.2"))

    ax = axes[1]
    means = [sc["n_sig_terms_panel_bg"].mean(), sc["n_sig_terms_genome_bg"].mean()]
    bars = ax.bar(["Panel-\nspecific", "Genome-\nwide"], means, color=[C_PANEL, C_GENOME], width=0.6)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 2, f"{m:.1f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Mean significant terms per list")
    ax.set_title("B. Mean across lists", loc="left", fontweight="bold")
    ax.set_ylim(0, max(means) * 1.25)

    fig.suptitle("Figure 1. Background choice alone determines whether any pathway reaches significance",
                 fontsize=11, fontweight="bold", y=1.03)
    fig.savefig(FIGDIR / "Figure1_H1_baseline.png")
    plt.close(fig)

    out = sc[["paper_id", "list_id", "species", "panel_size", "n_de_genes",
              "n_terms_tested_panel_bg", "n_terms_tested_genome_bg",
              "n_sig_terms_panel_bg", "n_sig_terms_genome_bg"]]
    out.to_csv(TABDIR / "Table2_step0_scored_lists.csv", index=False)
    print(f"Figure 1 + Table 2: {len(sc)} scored lists, "
          f"panel mean {means[0]:.3f}, genome mean {means[1]:.1f}")


def fig2_synthetic():
    """Fix 8: pure math - only background size N varies, everything else frozen."""
    d = pd.read_csv(ENR / "fix8_synthetic_simulation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

    ax = axes[0]
    ax.plot(d["background_size_N"], d["pvalue"], "o-", color=C_GENOME, lw=1.8, ms=5)
    ax.axhline(0.05, ls="--", color=C_NEUTRAL, lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(top=d["pvalue"].max() * 60)
    ax.text(0.97, 0.90, "p = 0.05", fontsize=8, color=C_NEUTRAL,
            transform=ax.transAxes, ha="right", va="bottom")
    ax.set_xlabel("Background size N (genes)")
    ax.set_ylabel("Enrichment p-value")
    ax.set_title("A. p-value vs. background size", loc="left", fontweight="bold")

    ax = axes[1]
    ax.plot(d["background_size_N"], d["fold_enrichment"], "s-", color=C_GENOME, lw=1.8, ms=5)
    ax.axhline(1.0, ls="--", color=C_NEUTRAL, lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Background size N (genes)")
    ax.set_ylabel("Fold enrichment")
    ax.set_title("B. Effect size vs. background size", loc="left", fontweight="bold")

    fig.suptitle("Figure 2. Synthetic control: pathway size, DE list size and true overlap held exactly fixed\n"
                 "(50-gene pathway, 40-gene DE list, 15-gene overlap at every N)",
                 fontsize=10, fontweight="bold", y=1.10)
    fig.savefig(FIGDIR / "Figure2_synthetic_control.png")
    plt.close(fig)
    d.to_csv(TABDIR / "Table3_synthetic_simulation.csv", index=False)
    print(f"Figure 2 + Table 3: p {d['pvalue'].iloc[0]:.3f} -> {d['pvalue'].iloc[-1]:.2e}, "
          f"fold {d['fold_enrichment'].iloc[0]}x -> {d['fold_enrichment'].iloc[-1]}x")


def fig3_doseresponse():
    """Fix 1 + Fix 9: controlled panel-size subsampling on two independent papers."""
    p03 = pd.read_csv(ENR / "fix1_paper03_subsampling_reactome_summary.csv").sort_values("panel_size")
    p07 = pd.read_csv(ENR / "fix1_paper07b_subsampling_reactome_summary.csv").sort_values("panel_size")
    p08 = pd.read_csv(ENR / "fix9_paper08_subsampling_reactome_summary.csv").sort_values("panel_size")
    d03 = pd.read_csv(ENR / "fix1_paper03_subsampling_reactome_draws.csv")
    d07 = pd.read_csv(ENR / "fix1_paper07b_subsampling_reactome_draws.csv")
    d08 = pd.read_csv(ENR / "fix9_paper08_subsampling_reactome_draws.csv")

    fig, axes = plt.subplots(3, 2, figsize=(9.5, 10.2))

    for row, (summ, draws, label, npanel) in enumerate([
        (p03, d03, "Paper 03 (950-gene CosMx panel)", 950),
        (p07, d07, "Paper 07b (398-gene MERFISH panel)", 398),
        (p08, d08, "Paper 08 (496-gene MERFISH panel)", 496),
    ]):
        ax = axes[row][0]
        # categorical x positions: the size ladder is not evenly spaced (900..25), so a
        # linear axis crowds the small-panel end and overlaps its tick labels
        sizes_desc = list(summ.sort_values("panel_size", ascending=False)["panel_size"])
        srt = summ.sort_values("panel_size", ascending=False)
        xpos = np.arange(len(sizes_desc))
        ax.errorbar(xpos, srt["mean_neglog10_pvalue_gap"],
                    yerr=srt["std_neglog10_pvalue_gap"], fmt="o-", color=C_GENOME,
                    lw=1.8, ms=5, capsize=3)
        ax.set_xticks(xpos)
        ax.set_xticklabels([str(int(v)) for v in sizes_desc])
        ax.set_xlim(-0.5, len(xpos) - 0.5)
        ax.set_xlabel("Panel size (genes retained)")
        ax.set_ylabel("Inflation gap\n(-log10 p, genome - panel)")
        ax.set_title(f"{'ACE'[row]}. {label}\nGap grows as panel shrinks", loc="left", fontweight="bold")

        ax = axes[row][1]
        # NaN handling: at the smallest panel sizes a random draw can leave ZERO
        # pathways passing the min/max term-size filter, so min_pvalue is undefined for
        # that draw. matplotlib silently drops an entire box if its series contains any
        # NaN, which previously made the 25-gene box vanish without warning. Drop the
        # NaNs explicitly and report how many draws were untestable - that count is
        # itself a result (a panel this small can fail to support any test at all).
        sizes_desc_b = sorted(summ["panel_size"], reverse=True)
        raw_parts = [draws.loc[draws["panel_size"] == s, "min_pvalue_panel_bg"].values
                     for s in sizes_desc_b]
        n_undefined = [int(np.isnan(v).sum()) for v in raw_parts]
        parts = [v[~np.isnan(v)] for v in raw_parts]
        pos = np.arange(len(parts))
        bp = ax.boxplot(parts, positions=pos, widths=0.6, patch_artist=True,
                        medianprops=dict(color="black"))
        for b in bp["boxes"]:
            b.set_facecolor(C_PANEL); b.set_alpha(0.55)
        ax.set_xticks(pos)
        ax.set_xticklabels([str(int(s)) for s in sorted(summ["panel_size"], reverse=True)])
        ax.set_xlim(-0.7, len(pos) - 0.3)
        ax.axhline(0.05, ls="--", color=C_NEUTRAL, lw=1)
        ax.set_xlabel("Panel size (genes retained)")
        ax.set_ylabel("Best p-value under\npanel background")
        for xi, nu in enumerate(n_undefined):
            if nu:
                ax.annotate(f"{nu} draws\nuntestable", xy=(xi, ax.get_ylim()[1] * 0.02),
                            fontsize=6.5, ha="center", va="bottom", color=C_NEUTRAL)
        ax.set_title(f"{'BDF'[row]}. Spread across {int(summ['n_draws'].iloc[0])} random draws", loc="left", fontweight="bold")

    fig.suptitle("Figure 3. Controlled subsampling on three datasets: only panel size varies, DE gene list frozen",
                 fontsize=11, fontweight="bold", y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIGDIR / "Figure3_doseresponse.png")
    plt.close(fig)

    p03 = p03.assign(paper="03_CosMx_AmyloidPlaqueNiche")
    p07 = p07.assign(paper="07b_MERFISH_PU1")
    p08 = p08.assign(paper="08_MERFISH_TcellMyelin")
    pd.concat([p03, p07, p08]).to_csv(TABDIR / "Table4_subsampling_summary.csv", index=False)
    for nm, d in [("03", p03), ("07b", p07), ("08", p08)]:
        print(f"Figure 3 + Table 4: {nm} gap "
              f"{d['mean_neglog10_pvalue_gap'].iloc[-1]:.2f} -> {d['mean_neglog10_pvalue_gap'].iloc[0]:.2f} "
              f"({int(d['n_draws'].iloc[0])} draws)")


def fig4_negative_control():
    """Fix 2: junk (non-significant) gene lists through the same pipeline."""
    f2 = pd.read_csv(ENR / "fix2_negative_control_summary_reactome.csv")
    f2 = f2[~f2["skipped"]]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    ax = axes[0]
    means = [f2["n_false_hits_panel_bg"].mean(), f2["n_false_hits_genome_bg"].mean()]
    bars = ax.bar(["Panel-\nspecific", "Genome-\nwide"], means, color=[C_PANEL, C_GENOME], width=0.6)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.08, f"{m:.3f}", ha="center",
                fontsize=9, fontweight="bold")
    ax.set_ylabel("Mean false 'significant' terms per list")
    ax.set_title("A. False hits on genes known NOT to be signal", loc="left", fontweight="bold")

    ax = axes[1]
    fracs = [(f2["n_false_hits_panel_bg"] > 0).mean() * 100,
             (f2["n_false_hits_genome_bg"] > 0).mean() * 100]
    bars = ax.bar(["Panel-\nspecific", "Genome-\nwide"], fracs, color=[C_PANEL, C_GENOME], width=0.6)
    for b, m in zip(bars, fracs):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.7, f"{m:.1f}%", ha="center",
                fontsize=9, fontweight="bold")
    ax.set_ylabel("% of junk lists producing >= 1 false hit")
    ax.set_title("B. How often it happens at all", loc="left", fontweight="bold")

    fig.suptitle("Figure 4. Negative control: the inflation is mechanical, not rediscovered biology",
                 fontsize=11, fontweight="bold", y=1.04)
    fig.savefig(FIGDIR / "Figure4_negative_control.png")
    plt.close(fig)
    f2.to_csv(TABDIR / "Table5_negative_control.csv", index=False)
    print(f"Figure 4 + Table 5: panel {means[0]:.4f} vs genome {means[1]:.4f} mean false hits; "
          f"{fracs[1]:.1f}% of lists affected under genome background")


def fig5_detected_background():
    """Fix 3: does the field's recommended 'detected genes' middle ground help here?
    Now covering all 7 papers / 54 scored lists, not the original 2-paper spot check."""
    d = pd.read_csv(ENR / "fix3_detected_background_reactome.csv")
    d = d[d["skipped"] == False].copy()
    order = ["panel_specific", "detected_genes", "genome_wide"]
    nice = {"panel_specific": "Panel-\nspecific", "detected_genes": "Detected\ngenes",
            "genome_wide": "Genome-\nwide"}
    cols = {"panel_specific": C_PANEL, "detected_genes": C_DETECT, "genome_wide": C_GENOME}

    size = d.pivot_table(index="list_id", columns="background_type", values="background_size")
    pval = d.pivot_table(index="list_id", columns="background_type", values="min_pvalue")
    sig = d.pivot_table(index="list_id", columns="background_type", values="n_sig_terms_padj05")
    paper = d.groupby("list_id")["paper_id"].first()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # A: detected as % of panel, one point per list, grouped by paper
    ax = axes[0]
    pct = (100 * size["detected_genes"] / size["panel_specific"]).rename("pct")
    dfp = pd.concat([pct, paper], axis=1).sort_values("paper_id")
    papers = list(dict.fromkeys(dfp["paper_id"]))
    for i, pid in enumerate(papers):
        v = dfp.loc[dfp["paper_id"] == pid, "pct"]
        ax.scatter(np.full(len(v), i) + np.random.default_rng(0).normal(0, 0.06, len(v)),
                   v, s=28, color=C_DETECT, alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.set_xticks(range(len(papers)))
    ax.set_xticklabels([p.split("_")[0] for p in papers])
    ax.set_ylim(0, 108)
    ax.axhline(100, ls=":", color=C_NEUTRAL, lw=1)
    ax.set_xlabel("Paper")
    ax.set_ylabel("Detected genes as % of panel")
    ax.set_title("A. 'Detected' is nearly the whole panel", loc="left", fontweight="bold")

    # B: background sizes, all lists, log scale
    ax = axes[1]
    for i, o in enumerate(order):
        v = size[o].dropna()
        ax.scatter(np.full(len(v), i) + np.random.default_rng(1).normal(0, 0.07, len(v)),
                   v, s=26, color=cols[o], alpha=0.7, edgecolor="white", linewidth=0.4)
    ax.set_yscale("log")
    ax.set_xticks(range(3)); ax.set_xticklabels([nice[o] for o in order])
    ax.set_ylabel("Background size (genes, log)")
    ax.set_title("B. Sizes across all 54 lists", loc="left", fontweight="bold")

    # C: resulting best p-value
    ax = axes[2]
    for i, o in enumerate(order):
        v = pval[o].dropna()
        ax.scatter(np.full(len(v), i) + np.random.default_rng(2).normal(0, 0.07, len(v)),
                   v, s=26, color=cols[o], alpha=0.7, edgecolor="white", linewidth=0.4)
    ax.set_yscale("log")
    ax.axhline(0.05, ls="--", color=C_NEUTRAL, lw=1)
    ax.text(0.99, 0.02, "p = 0.05", fontsize=7.5, color=C_NEUTRAL,
            transform=ax.transAxes, ha="right", va="bottom")
    ax.set_xticks(range(3)); ax.set_xticklabels([nice[o] for o in order])
    ax.set_ylabel("Best p-value obtained (log)")
    ax.set_title("C. Detected tracks panel, not genome", loc="left", fontweight="bold")

    fig.suptitle("Figure 5. Across all 7 studies, the field's recommended 'detected genes' background\n"
                 "behaves like the panel, not like a midpoint between panel and genome-wide",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(FIGDIR / "Figure5_detected_background.png")
    plt.close(fig)
    d.to_csv(TABDIR / "Table6_detected_background.csv", index=False)

    beat = int((pval["detected_genes"] < pval["panel_specific"]).sum())
    beatg = int((pval["genome_wide"] < pval["panel_specific"]).sum())
    print(f"Figure 5 + Table 6: {len(size)} scored lists across 7 papers | "
          f"detected/panel median {pct.median():.1f}% | "
          f"detected beats panel on {beat}/{len(pval)}, genome beats panel on {beatg}/{len(pval)} | "
          f"mean sig: panel {sig['panel_specific'].mean():.3f}, "
          f"detected {sig['detected_genes'].mean():.3f}, genome {sig['genome_wide'].mean():.1f}")


def fig6_confounding():
    """Fix 6b: the across-study regression flips sign once confounds are adjusted for."""
    r = pd.read_csv(ENR / "fix6b_adjustment_regression_reactome.csv")
    ps = r[r["covariate"] == "panel_size"]
    unadj = ps[ps["model"].str.startswith("(a)")].iloc[0]
    adj = ps[ps["model"].str.startswith("(b)")].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))

    ax = axes[0]
    vals = [unadj["coefficient"], adj["coefficient"]]
    cols = [C_GENOME if v > 0 else C_PANEL for v in vals]
    bars = ax.bar(["Unadjusted\n(panel size only)", "Adjusted\n(+ species, + DE list size)"],
                  vals, color=cols, width=0.55)
    ax.axhline(0, color="black", lw=1)
    lim = max(abs(v) for v in vals) * 1.85
    ax.set_ylim(-lim, lim)
    for b, v, p in zip(bars, vals, [unadj["pvalue"], adj["pvalue"]]):
        va = "bottom" if v > 0 else "top"
        off = lim * 0.04 if v > 0 else -lim * 0.04
        ax.text(b.get_x() + b.get_width() / 2, v + off, f"{v:+.4f}\n(p = {p:.3f})",
                ha="center", va=va, fontsize=8.5, fontweight="bold")
    ax.set_ylabel("Regression coefficient for panel size")
    ax.set_title("A. Adjustment reveals the effect", loc="left", fontweight="bold", pad=10)
    ax.text(0.98, 0.97, "positive = larger panel, MORE inflation\n(would contradict H2)",
            transform=ax.transAxes, fontsize=7.5, va="top", ha="right", color=C_GENOME)
    ax.text(0.02, 0.03, "negative = smaller panel, MORE inflation\n(supports H2)",
            transform=ax.transAxes, fontsize=7.5, va="bottom", ha="left", color=C_PANEL)

    ax = axes[1]
    dat = pd.read_csv(ENR / "fix6b_adjustment_data_reactome.csv")
    ycol = [c for c in dat.columns if "gap" in c.lower()][0]
    spec_col = [c for c in dat.columns if "species" in c.lower()]
    if spec_col:
        # collapse verbose free-text species strings ("Mouse (5xFAD; PU.1-low...)") to Mouse/Human
        simple = dat[spec_col[0]].astype(str).str.contains("human", case=False)
        dat = dat.assign(_sp=np.where(simple, "Human", "Mouse"))
        for key, col in [("Mouse", C_GENOME), ("Human", C_PANEL)]:
            grp = dat[dat["_sp"] == key]
            if len(grp):
                ax.scatter(grp["panel_size"], grp[ycol], s=30, alpha=0.7,
                           color=col, edgecolor="white", linewidth=0.4,
                           label=f"{key} papers (n={len(grp)} lists)")
        ax.legend(fontsize=8, loc="upper right")
    else:
        ax.scatter(dat["panel_size"], dat[ycol], s=30, alpha=0.7, color=C_NEUTRAL)
    ax.set_xlabel("Panel size (genes)")
    ax.set_ylabel("Inflation gap (-log10 p)")
    ax.set_title("B. Why: the groups are not comparable", loc="left", fontweight="bold", pad=10)

    fig.suptitle("Figure 6. The naive 7-paper comparison is uninformative until confounders are adjusted for\n"
                 "(unadjusted: a flat null; adjusted: a strong effect in H2's direction)",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(FIGDIR / "Figure6_confounding.png")
    plt.close(fig)
    r.to_csv(TABDIR / "Table7_confound_regression.csv", index=False)
    print(f"Figure 6 + Table 7: unadjusted {unadj['coefficient']:+.4f} -> adjusted {adj['coefficient']:+.4f}")


def table1_datasets():
    """Table 1: the dataset/covariate table, cleaned up for publication."""
    cov = pd.read_csv(RES / "covariate_table.csv")
    cov.to_csv(TABDIR / "Table1_datasets_and_covariates.csv", index=False)
    print(f"Table 1: {len(cov)} papers")


if __name__ == "__main__":
    table1_datasets()
    fig1_baseline()
    fig2_synthetic()
    fig3_doseresponse()
    fig4_negative_control()
    fig5_detected_background()
    fig6_confounding()
    print(f"\nAll figures -> {FIGDIR}")
    print(f"All tables  -> {TABDIR}")
