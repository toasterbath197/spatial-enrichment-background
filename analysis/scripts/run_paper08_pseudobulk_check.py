"""
Section 3.8c follow-up: pseudobulk vs per-cell DE concordance for dataset 08
(MERFISH T cell neighbor vs non-neighbor, within 5xFAD, 5 tissue sections).

This dataset's grouping variable (T-cell spatial neighbor vs non-neighbor) is
WITHIN each section, not between animals - there is no "sample belongs to one
group" structure the way there is for datasets 05/06/07b. The natural
replication unit here is the section: each of the 5 sections contributes one
neighbor-pseudobulk point and one non-neighbor-pseudobulk point, so the two
arms are PAIRED by section. We therefore use a paired Wilcoxon signed-rank
test across the 5 sections, rather than the unpaired Mann-Whitney U used for
the animal-level datasets - the more appropriate test for a paired design, and
a deliberately different choice from pseudobulk_de.py's function, noted here
rather than silently reused.

Adapted from run_paper08_tcellmyelin.py's loading and neighbor-finding code.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import gc

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests
from de_pipeline import stats_utils, enrichment as en
from de_pipeline.paths import DATA_ROOT, DE_TABLES_DIR

EXPR = str(DATA_ROOT / "08_MERFISH_TcellMyelin/GSE243120_MERFISH_5xFAD_expression.txt.gz")
META = str(DATA_ROOT / "08_MERFISH_TcellMyelin/GSE243120_MERFISH_5xFAD_metadata.txt.gz")
PAPER_ID = "08_MERFISH_TcellMyelin"
K_NEIGHBORS = 50


def main():
    print(f"[{PAPER_ID}] loading metadata + expression ...")
    meta = pd.read_csv(META, sep="\t")
    header = pd.read_csv(EXPR, sep="\t", nrows=0).columns.tolist()
    gene_cols = [c for c in header if c != "cellID"]
    dtype_map = {c: "int32" for c in gene_cols}
    dtype_map["cellID"] = "int64"
    expr = pd.read_csv(EXPR, sep="\t", dtype=dtype_map)
    df = expr.merge(meta, on="cellID", how="inner")
    del expr, meta
    gc.collect()
    print(f"  merged: {len(df):,} cells, {len(gene_cols)} genes, {df['section'].nunique()} sections")

    df["group"] = None
    for sec, g in df.groupby("section"):
        is_tcell = g["annotation"] == "Tcell"
        tcell_xy = g.loc[is_tcell, ["center_x", "center_y"]].values
        other_idx = g.index[~is_tcell]
        other_xy = g.loc[other_idx, ["center_x", "center_y"]].values
        if len(tcell_xy) < 2 or len(other_idx) < K_NEIGHBORS:
            continue
        tree = cKDTree(other_xy)
        k = min(K_NEIGHBORS, len(other_idx))
        _, nn = tree.query(tcell_xy, k=k)
        neighbor_positions = np.unique(nn.ravel())
        neighbor_idx = other_idx[neighbor_positions]
        nonneighbor_idx = other_idx.difference(neighbor_idx)
        df.loc[neighbor_idx, "group"] = "Tcell_neighbor"
        df.loc[nonneighbor_idx, "group"] = "Tcell_nonneighbor"

    group_mask = df["group"].notna()
    labels = df.loc[group_mask, "group"].values
    sections = df.loc[group_mask, "section"].values
    raw = df.loc[group_mask, gene_cols].values.astype(np.float32)
    del df
    gc.collect()
    norm = stats_utils.normalize_cells(raw)
    print(f"  grouped cells: {len(labels):,} (neighbor={sum(labels=='Tcell_neighbor'):,}, "
          f"nonneighbor={sum(labels=='Tcell_nonneighbor'):,}) across {len(set(sections))} sections")

    # --- Tier A: per-cell (project's primary method for this dataset, Bonferroni) ---
    mask1 = labels == "Tcell_neighbor"
    mask2 = labels == "Tcell_nonneighbor"
    x1, x2 = norm[mask1], norm[mask2]
    n_genes = len(gene_cols)
    pvals = np.empty(n_genes, dtype=float)
    gene_chunk = 20
    for start in range(0, n_genes, gene_chunk):
        end = min(start + gene_chunk, n_genes)
        _, p_chunk = scipy_stats.mannwhitneyu(
            x1[:, start:end], x2[:, start:end], axis=0,
            alternative="two-sided", method="asymptotic")
        pvals[start:end] = p_chunk
    padj_cell = multipletests(pvals, method="bonferroni")[1]
    sig_cell = set(np.array(gene_cols)[padj_cell < 0.05])
    print(f"  per-cell: {len(sig_cell)}/{n_genes} significant (Bonferroni, matching primary method)")

    # --- Tier B: pseudobulk, one row per (section, group) ---
    pb_df = pd.DataFrame(norm, columns=gene_cols)
    pb_df["section"] = sections
    pb_df["group"] = labels
    pb = pb_df.groupby(["section", "group"]).mean()

    sections_with_both = [s for s in set(sections)
                           if (s, "Tcell_neighbor") in pb.index and (s, "Tcell_nonneighbor") in pb.index]
    print(f"  sections with both arms present: {len(sections_with_both)}")

    neighbor_pb = pb.loc[[(s, "Tcell_neighbor") for s in sections_with_both]].values
    nonneighbor_pb = pb.loc[[(s, "Tcell_nonneighbor") for s in sections_with_both]].values

    pb_pvals = np.full(n_genes, np.nan)
    for g in range(n_genes):
        try:
            _, p = scipy_stats.wilcoxon(neighbor_pb[:, g], nonneighbor_pb[:, g])
        except ValueError:
            p = np.nan  # all differences zero, or too few sections
        pb_pvals[g] = p
    mask_valid = ~np.isnan(pb_pvals)
    padj_pb = np.full(n_genes, np.nan)
    if mask_valid.sum():
        padj_pb[mask_valid] = multipletests(pb_pvals[mask_valid], method="bonferroni")[1]
    sig_pb = set(np.array(gene_cols)[np.nan_to_num(padj_pb, nan=1.0) < 0.05])
    n_pb_testable = int(mask_valid.sum())
    print(f"  pseudobulk (paired, {len(sections_with_both)} sections, Wilcoxon signed-rank, "
          f"Bonferroni): {len(sig_pb)}/{n_pb_testable} testable significant")

    de_cell = pd.DataFrame({"gene": gene_cols, "pvalue": pvals, "padj": padj_cell})
    de_pb = pd.DataFrame({"gene": gene_cols, "pvalue": pb_pvals, "padj": padj_pb})
    de_cell.to_csv(DE_TABLES_DIR / "paper08_pseudobulk_check_percell.csv", index=False)
    de_pb.to_csv(DE_TABLES_DIR / "paper08_pseudobulk_check_pseudobulk.csv", index=False)

    merged = de_cell.merge(de_pb, on="gene", suffixes=("_cell", "_pb")).dropna()
    if len(merged) > 2:
        rho, rho_p = scipy_stats.spearmanr(merged["pvalue_cell"], merged["pvalue_pb"])
        print(f"  Spearman rho(p_cell, p_pseudobulk) = {rho:.3f} (p={rho_p:.2e}), n={len(merged)} genes")

    panel = gene_cols
    gmt, universe = en.load_gmt("mouse"), en.gmt_universe("mouse")
    for label, query in (("per-cell", sig_cell), ("pseudobulk", sig_pb)):
        if len(query) < 3:
            print(f"  {label}: too few DE genes ({len(query)}) to test enrichment")
            continue
        panel_res = en.hypergeometric_enrichment(query, panel, gmt)
        genome_res = en.hypergeometric_enrichment(query, universe, gmt)
        n_panel_sig = int((panel_res["padj"] < 0.05).sum()) if len(panel_res) else 0
        n_genome_sig = int((genome_res["padj"] < 0.05).sum()) if len(genome_res) else 0
        print(f"  {label} DE list -> enrichment: panel_sig={n_panel_sig}, genome_sig={n_genome_sig}")


if __name__ == "__main__":
    main()
