"""
Paper 8 - MERFISH T cell / Myelin Pathology (5xFAD) - self-computed.

CORRECTED 2026-08-03 (group definition): earlier assumed this paper needed a
WT arm and marked it blocked because GSE243120 only has 5xFAD sections. Wrong
- the target comparison is the paper's own Figure 2h: T-cell spatial
neighbors vs. non-neighbors, entirely WITHIN the 5xFAD mice. The paper's
WT-vs-5xFAD arm used a separate scRNA-seq dataset (GSE243018), out of scope
here.

REWRITTEN 2026-08-04 (exact method, per updated 08_MERFISH_TcellMyelin/
README.txt, which quotes the paper's own Methods verbatim): the original
version of this script approximated proximity with a quartile split on
distance-to-nearest-T-cell, restricted to 'Immune'-annotated cells only. The
paper's actual method is different on both counts:

  "we calculated 50 nearest neighbors of every T cell based on
  two-dimensional spatial coordinates of cell centroids. We then identified
  differentially expressed genes between T cell neighbors and the remaining
  cells (nonneighbors) using the FindMarkers function with a Wilcoxon
  rank-sum test." (Bonferroni-adjusted, per the Fig 2h legend.)

So: neighbors = union of each T cell's 50 nearest OTHER cells (any annotation,
not just Immune), computed per section (center_x/center_y are section-local).
Non-neighbors = every other non-T-cell in the same sections. T cells
themselves are excluded from both groups - the DE question is about the
surrounding tissue's response, not the T cells. This is a case where we
replicate the paper's own group definition and correction method exactly
(Bonferroni), rather than the project's usual standardized primary method
(Wilcoxon is the same either way; the correction differs) - documented
deliberately, see 00_PROJECT_NOTES_AND_METHODOLOGY.txt section 2 and this
paper's README for why.

Confirmed: metadata has exactly 93 'Tcell'-annotated cells total across the 5
sections, matching the paper's own figure legend ("93 CD8+ T cells").
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import gc

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests
from de_pipeline import datasets, schema, stats_utils
from de_pipeline.paths import DATA_ROOT, DE_TABLES_DIR


EXPR = str(DATA_ROOT / "08_MERFISH_TcellMyelin/GSE243120_MERFISH_5xFAD_expression.txt.gz")
META = str(DATA_ROOT / "08_MERFISH_TcellMyelin/GSE243120_MERFISH_5xFAD_metadata.txt.gz")
PAPER_ID = "08_MERFISH_TcellMyelin"
K_NEIGHBORS = 50


def main():
    print(f"[{PAPER_ID}] loading metadata + expression ...")
    meta = pd.read_csv(META, sep="\t")
    print(f"  {len(meta):,} total cells, {(meta['annotation'] == 'Tcell').sum()} Tcells, "
          f"{meta['section'].nunique()} sections")

    header = pd.read_csv(EXPR, sep="\t", nrows=0).columns.tolist()
    gene_cols = [c for c in header if c != "cellID"]
    dtype_map = {c: "int32" for c in gene_cols}
    dtype_map["cellID"] = "int64"
    expr = pd.read_csv(EXPR, sep="\t", dtype=dtype_map)
    df = expr.merge(meta, on="cellID", how="inner")
    del expr, meta
    gc.collect()
    print(f"  merged: {len(df):,} cells, {len(gene_cols)} genes")

    df["group"] = None
    for sec, g in df.groupby("section"):
        is_tcell = g["annotation"] == "Tcell"
        tcell_xy = g.loc[is_tcell, ["center_x", "center_y"]].values
        other_idx = g.index[~is_tcell]
        other_xy = g.loc[other_idx, ["center_x", "center_y"]].values
        if len(tcell_xy) < 2 or len(other_idx) < K_NEIGHBORS:
            print(f"  [skip section {sec}] too few Tcells/other cells")
            continue
        tree = cKDTree(other_xy)
        k = min(K_NEIGHBORS, len(other_idx))
        _, nn = tree.query(tcell_xy, k=k)  # indices into other_idx/other_xy, per Tcell
        neighbor_positions = np.unique(nn.ravel())
        neighbor_idx = other_idx[neighbor_positions]
        nonneighbor_idx = other_idx.difference(neighbor_idx)
        df.loc[neighbor_idx, "group"] = "Tcell_neighbor"
        df.loc[nonneighbor_idx, "group"] = "Tcell_nonneighbor"
        print(f"  section {sec}: {len(tcell_xy)} Tcells, {len(neighbor_idx):,} neighbor cells "
              f"(k={k}), {len(nonneighbor_idx):,} non-neighbor cells")

    group_mask = df["group"].notna()
    labels = df.loc[group_mask, "group"].values
    raw = df.loc[group_mask, gene_cols].values.astype(np.float32)
    del df
    gc.collect()
    norm = stats_utils.normalize_cells(raw)
    print(f"  total grouped cells: {len(labels):,} "
          f"(neighbor={sum(labels=='Tcell_neighbor'):,}, "
          f"nonneighbor={sum(labels=='Tcell_nonneighbor'):,})")

    # Wilcoxon rank-sum, but Bonferroni-corrected here specifically (not the
    # project's default BH) to match the paper's own Fig 2h legend exactly -
    # see module docstring.
    mask1 = labels == "Tcell_neighbor"
    mask2 = labels == "Tcell_nonneighbor"
    x1, x2 = norm[mask1], norm[mask2]
    n_genes = x1.shape[1]
    pvals = np.empty(n_genes, dtype=float)
    gene_chunk = 20
    for start in range(0, n_genes, gene_chunk):
        end = min(start + gene_chunk, n_genes)
        _, p_chunk = scipy_stats.mannwhitneyu(
            x1[:, start:end], x2[:, start:end], axis=0,
            alternative="two-sided", method="asymptotic",
        )
        pvals[start:end] = p_chunk
    padj = multipletests(pvals, method="bonferroni")[1]
    pseudocount = 1e-9
    mean1 = np.expm1(x1).mean(axis=0)
    mean2 = np.expm1(x2).mean(axis=0)
    log2fc = np.log2(mean1 + pseudocount) - np.log2(mean2 + pseudocount)

    de = pd.DataFrame({
        "gene": gene_cols, "log2FC": log2fc, "pvalue": pvals, "padj": padj,
        "n_group1": int(mask1.sum()), "n_group2": int(mask2.sum()),
    })
    de["cell_type"] = "all"
    de["comparison_type"] = "T-cell neighbors (50-NN) vs non-neighbors, within 5xFAD (Fig 2h replication)"
    de["list_id"] = f"{PAPER_ID}__Tcell_neighbor_vs_nonneighbor__all"
    datasets.stamp(de, PAPER_ID)

    out = schema.make_df(de.to_dict("records"))
    outpath = str(DE_TABLES_DIR / "08_MERFISH_TcellMyelin_DE.csv")
    schema.write_table(out, outpath)
    return out


if __name__ == "__main__":
    main()
