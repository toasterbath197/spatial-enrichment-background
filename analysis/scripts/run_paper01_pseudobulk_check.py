"""
Section 3.8c follow-up: pseudobulk vs per-cell DE concordance for dataset 01
(Trem2 R47H MERFISH, mouse, 9 cell types x 5 genotype comparisons = 45 lists
-- the dominant dataset in the baseline corpus, 45 of the 54 total scored
lists).

This dataset's 5.5 GB h5ad could not be staged into the cloud analysis
environment (400 MB per-file cap), and the local device-bridge sandbox that
already holds the file had no h5py, no scipy, no compiler, and confirmed no
general internet access. See ADDITIONS_NOT_FROM_RESULTS.md ("1c. Addendum",
29 Aug 2026) and 00_PROJECT_NOTES_AND_METHODOLOGY.txt (Section 42,
"DATASET 01: INVESTIGATED AND RULED OUT") for the full account of that
blocker, and "1d. Addendum" / Section 43 (30 Aug 2026) for how it was
resolved: a platform-matched h5py wheel
(h5py-3.14.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl,
downloaded in the cloud container -- which has internet access -- pushed to
the device under _vendor/, and installed with
`pip install --no-index --no-deps _vendor/h5py-*.whl`) made h5py available
on-device without needing any network access there. scipy and statsmodels
are still NOT available on-device, so this file is deliberately split into
two stages that run in different places:

  STAGE 1 (aggregate_pseudobulk(), run via device_bash on the machine
  holding MERFISH_Data.h5ad): pure numpy/pandas plus the newly-installed
  h5py only -- no scipy, no statsmodels, no repo package import, so it can
  run standalone on a minimal sandbox. Produces
  analysis/results/DE_tables/paper01_pseudobulk_matrix.csv: one row per
  (cell_type, region), ~150 rows x 300 genes -- small enough to transfer
  back to the cloud environment.

  STAGE 2 (compute_stats(), run in the cloud analysis environment or
  anywhere with scipy/statsmodels/pandas): joins the Stage-1 matrix against
  the per-cell results already in master_DE_table.csv, runs Mann-Whitney U
  + BH per gene per (cell_type, comparison), and reports the significance
  counts plus Spearman rho between the two tests' p-values.

Adapted from run_paper01_trem2.py's loading/cell-type/comparison logic
(same CELL_TYPES and COMPARISONS lists). Uses obs['sample'] as the
mouse/region-level replicate identifier -- confirmed usable this round: of
18 distinct regions, 17 are genotype-pure by obs['gen_coarse'] and 1 mixes
genotypes and is excluded rather than assigned to either side. This answers
the "NOT YET CONFIRMED TO EXIST" open question flagged against obs['sample']
in the prior save point.

VERIFIED (30 Aug 2026): re-running Stage 2 against the already-saved
paper01_pseudobulk_matrix.csv reproduces the saved
paper01_pseudobulk_check_summary.csv's n1, n2, n_sig_percell and
n_sig_pseudobulk columns exactly (all 45 rows) -- this is the headline
result (mean 75.1% per-cell "significant" vs 0/45 pseudobulk-significant)
and it is exactly reproducible. The Spearman rho column is NOT exactly
reproduced by this reconstruction (this file gives a non-null rho for all
45 rows with mean ~0.47; the saved summary has non-null rho for 35/45 rows
with mean ~0.485) -- the original inline computation that produced the
saved file was not preserved verbatim, and some environment- or
code-path-specific detail in how ties/NaN p-values were handled differs.
Flagged here rather than silently smoothed over: the significance counts
that every manuscript conclusion rests on are solid; the correlation
coefficient, a secondary/supporting statistic, is reproducible only
approximately from this script.
"""
import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# STAGE 1 - run via device_bash, on the machine holding MERFISH_Data.h5ad.
# Needs: numpy, pandas, h5py (only after the offline wheel install above).
# Does NOT need scipy/statsmodels/the de_pipeline package.
# ---------------------------------------------------------------------------

CELL_TYPES = ["AST", "Amygdala ExN", "Cortical ExN", "Cortical Inh", "Hippocampal ExN",
              "Mic", "OGC", "Other Glia", "Subcortical Neurons"]


def normalize_cells(X, target_sum=None):
    """Reimplemented inline (matches de_pipeline.stats_utils.normalize_cells
    exactly) so Stage 1 has zero dependency on the repo package -- it needs
    to run standalone on a machine that may not have it on the path."""
    X = np.asarray(X, dtype=np.float32)
    totals = X.sum(axis=1, dtype=np.float64)
    totals = np.where(totals == 0, 1.0, totals).astype(np.float32)
    if target_sum is None:
        target_sum = np.float32(np.median(totals))
    X *= (target_sum / totals)[:, None]
    np.log1p(X, out=X)
    return X


def _read_cat(f, name):
    import h5py
    grp = f["obs/" + name]
    if isinstance(grp, h5py.Group):
        cats = [c.decode() if isinstance(c, bytes) else c for c in grp["categories"][:]]
        codes = grp["codes"][:]
        return np.array([cats[c] if c >= 0 else None for c in codes], dtype=object)
    return grp[:]


def aggregate_pseudobulk(h5ad_path, out_csv):
    import h5py
    with h5py.File(h5ad_path, "r") as f:
        genes = [x.decode() if isinstance(x, bytes) else x for x in f["var/_index"][:]]
        sample = _read_cat(f, "sample")
        gen_coarse = _read_cat(f, "gen_coarse")
        cluster_coarse = _read_cat(f, "cluster_coarse")

        # Chunked read straight into a preallocated float32 array: the raw
        # layer is ~1 GB as float64, and holding both the float64 source and
        # a float32 copy at once does not fit the sandbox's ~3.8 GB RAM.
        n_obs, n_var = f["layers/RNA"].shape
        raw = np.empty((n_obs, n_var), dtype=np.float32)
        ds = f["layers/RNA"]
        chunk = 50_000
        for start in range(0, n_obs, chunk):
            end = min(start + chunk, n_obs)
            raw[start:end] = ds[start:end].astype(np.float32)

    # obs['sample'] looks like ".../region_N[/...]" -- collapse to the
    # region-level identifier and treat each region as one replicate.
    short = pd.Series(sample).str.extract(r'([^/]+/region_\d+)')[0].values

    # A region should belong to exactly one genotype; drop any that mix
    # genotypes rather than silently assign it to one side.
    df_check = pd.DataFrame({"region": short, "gen_coarse": gen_coarse}).drop_duplicates()
    n_unique = df_check["region"].nunique()
    mixed_counts = df_check.groupby("region")["gen_coarse"].nunique()
    mixed_regions = set(mixed_counts[mixed_counts > 1].index)
    keep_mask = ~pd.Series(short).isin(mixed_regions).values
    print(f"  regions: {n_unique} total, {len(mixed_regions)} mixed-genotype (excluded)")

    rows = []
    for ct in CELL_TYPES:
        mask = (cluster_coarse == ct) & keep_mask
        if mask.sum() < 50:
            continue
        sub_norm = normalize_cells(raw[mask])
        sub_region = short[mask]
        sub_gen = gen_coarse[mask]
        pb_df = pd.DataFrame(sub_norm, columns=genes)
        pb_df["region"] = sub_region
        pb = pb_df.groupby("region").mean()
        region_gen = pd.Series(sub_gen, index=sub_region).groupby(level=0).first()
        for region_id, row in pb.iterrows():
            rows.append({"cell_type": ct, "region": region_id,
                         "genotype": region_gen[region_id], **row.to_dict()})
        print(f"  {ct}: {int(mask.sum()):,} cells -> {len(pb)} region pseudobulk rows")

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv} ({out.shape[0]} rows x {out.shape[1]} cols)")
    return out


# ---------------------------------------------------------------------------
# STAGE 2 - run wherever scipy/statsmodels/pandas are available (the cloud
# analysis environment, in this project's case). Consumes only the small
# Stage-1 output CSV, not the 5.5 GB h5ad.
# ---------------------------------------------------------------------------

COMPARISONS = [("5xFAD", "WT"), ("Trem2", "WT"), ("Trem2_5xFAD", "Trem2"),
               ("Trem2_5xFAD", "5xFAD"), ("Trem2_5xFAD", "WT")]


def compute_stats(pseudobulk_csv, master_de_csv, out_summary_csv):
    from scipy import stats as scipy_stats
    from scipy.stats import spearmanr
    from statsmodels.stats.multitest import multipletests

    pb = pd.read_csv(pseudobulk_csv)
    master = pd.read_csv(master_de_csv)
    master = master[master.paper_id == "01_Trem2R47H_MERFISH"]
    gene_cols = [c for c in pb.columns if c not in ("cell_type", "region", "genotype")]

    rows = []
    for ct, g in pb.groupby("cell_type"):
        for g1, g2 in COMPARISONS:
            x1 = g.loc[g.genotype == g1, gene_cols].values
            x2 = g.loc[g.genotype == g2, gene_cols].values
            if x1.shape[0] < 2 or x2.shape[0] < 2:
                continue
            pvals = np.empty(len(gene_cols))
            for j in range(len(gene_cols)):
                try:
                    _, p = scipy_stats.mannwhitneyu(x1[:, j], x2[:, j], alternative="two-sided")
                except ValueError:
                    p = np.nan
                pvals[j] = p
            valid = ~np.isnan(pvals)
            padj = np.full(len(gene_cols), np.nan)
            if valid.sum():
                padj[valid] = multipletests(pvals[valid], method="fdr_bh")[1]
            n_sig_pb = int((np.nan_to_num(padj, nan=1.0) < 0.05).sum())

            percell = master[(master.cell_type == ct) & (master.comparison_type == f"{g1} vs {g2}")]
            n_sig_cell = int((percell["padj"] < 0.05).sum())
            n_total = len(percell)

            rho = np.nan
            merged_p = percell.set_index("gene")["pvalue"].reindex(gene_cols)
            pv_series = pd.Series(pvals, index=gene_cols)
            valid_pair = merged_p.notna() & pv_series.notna()
            if valid_pair.sum() > 2:
                rho, _ = spearmanr(merged_p[valid_pair], pv_series[valid_pair])

            rows.append({"cell_type": ct, "comparison": f"{g1}_vs_{g2}",
                         "n1": x1.shape[0], "n2": x2.shape[0],
                         "n_sig_percell": n_sig_cell, "n_total": n_total,
                         "n_sig_pseudobulk": n_sig_pb, "rho": rho})

    res = pd.DataFrame(rows)
    res.to_csv(out_summary_csv, index=False)
    print(f"  wrote {out_summary_csv} ({len(res)} rows)")
    pct = res.n_sig_percell / res.n_total * 100
    print(f"  mean per-cell sig: {pct.mean():.1f}% ({pct.min():.1f}-{pct.max():.1f}%)")
    print(f"  pseudobulk sig: {int(res.n_sig_pseudobulk.sum())}/{len(res)}")
    print(f"  mean rho: {res.rho.dropna().mean():.3f} (n={int(res.rho.notna().sum())})")
    return res


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or sys.argv[1] not in ("aggregate", "analyze"):
        print("usage: run_paper01_pseudobulk_check.py {aggregate|analyze}")
        print("  aggregate: run on-device (needs h5py, has the h5ad file)")
        print("  analyze:   run in the cloud env (needs scipy/statsmodels/master_DE_table.csv)")
        sys.exit(1)

    if sys.argv[1] == "aggregate":
        ROOT = os.path.expanduser("~/mnt/failure enrichment analysis")
        aggregate_pseudobulk(
            os.path.join(ROOT, "01_Trem2R47H_MERFISH/MERFISH_Data.h5ad"),
            os.path.join(ROOT, "analysis/results/DE_tables/paper01_pseudobulk_matrix.csv"),
        )
    else:
        import pathlib
        HERE = pathlib.Path(__file__).resolve().parents[1]
        compute_stats(
            HERE / "results/DE_tables/paper01_pseudobulk_matrix.csv",
            HERE / "results/DE_tables/master_DE_table.csv",
            HERE / "results/DE_tables/paper01_pseudobulk_check_summary.csv",
        )
