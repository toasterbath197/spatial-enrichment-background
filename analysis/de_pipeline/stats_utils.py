"""
Generic, platform-agnostic differential expression helpers.

PRIMARY (standardized) method - see 00_PROJECT_NOTES_AND_METHODOLOGY.txt
section 2. DE-calling method must NOT vary between papers, or it becomes a
confound entangled with the actual variable of interest (background gene set
choice). So every paper's primary DE list is generated the same way:

  normalize_cells()
      Total-count normalize each cell to the across-dataset median total
      count, then log1p. Applied to each paper's RAW counts (not whatever
      normalization, if any, the original authors' own object happened to
      ship with) so the "normalized counts" input to the test is defined
      identically everywhere.

  wilcoxon_two_group_de()
      Per-gene Wilcoxon rank-sum (Mann-Whitney U) test between two groups of
      CELLS (not pseudobulk) on normalize_cells() output - matches Seurat's
      FindMarkers(test.use="wilcox") default, which is also what several of
      the source papers' own authors used (e.g. 7b, 8), so our primary tier
      and those papers' own numbers are at least directly comparable in
      method. BH-corrected (project's fixed correction per the pre-analysis
      plan) - note several source papers used Bonferroni for their own
      figures; that's a deliberate, documented difference, not an oversight.

SECONDARY / exception paths:

  continuous_covariate_de()
      Per-gene Spearman correlation between a continuous covariate (e.g.
      pseudoprogression score) and expression, on pseudobulk units (avoids
      pseudoreplication - many cells per donor are not independent draws of
      the donor-level covariate). This is paper 4's own necessary code path
      (continuous CPS score, not a discrete group comparison) - explicitly
      exempted from the "one standardized method" rule in the project notes,
      not a shortcut.

  build_pseudobulk()
      Aggregates cells to one row per donor/sample, used by the above.

An earlier pseudobulk Welch t-test lived here as well. Every paper was moved off
it in the 2026-08-04 two-tier redesign (project notes section 2), leaving it
called by nothing, so it was removed rather than left in the repository looking
like part of the method. The history is in the notes, not in dead code.

None of these assume a particular platform, panel size, or organism - they
operate on plain (samples x genes) or (cells x genes) numeric matrices.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def _bh(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan)
    mask = ~np.isnan(pvals)
    if mask.sum() > 0:
        out[mask] = multipletests(pvals[mask], method="fdr_bh")[1]
    return out


def normalize_cells(X: np.ndarray, target_sum: float | None = None) -> np.ndarray:
    """Total-count normalize each cell (row) to `target_sum` (defaults to the
    median per-cell total in X), then log1p. X must be RAW counts - always
    pull each paper's raw layer/matrix and normalize here, rather than reusing
    whatever normalization (if any) an author's object already carries, so
    every paper's Wilcoxon test runs on identically-defined input.

    Stays in float32 throughout (in place where possible) - this project's
    analysis environment has very little RAM, and float32 precision is ample
    for log-normalized expression values feeding a rank-based test.
    """
    X = np.asarray(X, dtype=np.float32)
    totals = X.sum(axis=1, dtype=np.float64)  # sum in float64 to avoid overflow, cast back down
    totals = np.where(totals == 0, 1.0, totals).astype(np.float32)
    if target_sum is None:
        target_sum = np.float32(np.median(totals))
    X *= (target_sum / totals)[:, None]  # in-place scale
    np.log1p(X, out=X)                    # in-place log1p
    return X


def wilcoxon_two_group_de(norm_expr: np.ndarray, gene_names, cell_group: np.ndarray,
                           group1: str, group2: str, min_cells: int = 10,
                           pseudocount: float = 1e-9, gene_chunk: int = 20) -> pd.DataFrame:
    """Per-gene Wilcoxon rank-sum test between two groups of cells (rows) of
    normalize_cells() output. log2FC follows the Seurat avg_log2FC convention:
    back-transform each group's mean log-normalized expression to linear scale
    (expm1), average, THEN take log2 of the ratio - not a raw difference of
    already-log values.

    Genes are processed in chunks of `gene_chunk` - scipy's vectorized
    mannwhitneyu(axis=0) internally builds full-size rank/tie-correction
    arrays, and with cell counts in the hundreds of thousands doing all genes
    at once reliably OOMs this project's memory-constrained analysis
    environment. Chunking bounds peak memory with no change to the result.

    Returns DataFrame(gene, log2FC, pvalue, padj, n_group1, n_group2).
    """
    cell_group = np.asarray(cell_group)
    mask1 = cell_group == group1
    mask2 = cell_group == group2
    n1, n2 = int(mask1.sum()), int(mask2.sum())
    if n1 < min_cells or n2 < min_cells:
        raise ValueError(
            f"Need >={min_cells} cells per group for a Wilcoxon test; "
            f"got n({group1})={n1}, n({group2})={n2}"
        )
    x1 = norm_expr[mask1]
    x2 = norm_expr[mask2]
    n_genes = x1.shape[1]
    pvals = np.empty(n_genes, dtype=float)
    for start in range(0, n_genes, gene_chunk):
        end = min(start + gene_chunk, n_genes)
        _, p_chunk = stats.mannwhitneyu(
            x1[:, start:end], x2[:, start:end], axis=0,
            alternative="two-sided", method="asymptotic",
        )
        pvals[start:end] = p_chunk
    mean1 = np.expm1(x1).mean(axis=0)
    mean2 = np.expm1(x2).mean(axis=0)
    log2fc = np.log2(mean1 + pseudocount) - np.log2(mean2 + pseudocount)
    padj = _bh(pvals)
    return pd.DataFrame({
        "gene": gene_names,
        "log2FC": log2fc,
        "pvalue": pvals,
        "padj": padj,
        "n_group1": n1,
        "n_group2": n2,
    })


def build_pseudobulk(expr: np.ndarray, gene_names, sample_ids: np.ndarray,
                      agg: str = "mean") -> pd.DataFrame:
    """Collapse a (cells x genes) matrix into (samples x genes) pseudobulk.

    expr: 2D array-like, cells x genes (dense or will be densified)
    sample_ids: 1D array of length n_cells giving each cell's sample/replicate id
    Returns a DataFrame indexed by sample_id, columns = gene_names.
    """
    expr = np.asarray(expr)
    df = pd.DataFrame(expr, columns=gene_names)
    df["__sample__"] = np.asarray(sample_ids)
    if agg == "mean":
        pb = df.groupby("__sample__").mean()
    elif agg == "sum":
        pb = df.groupby("__sample__").sum()
    else:
        raise ValueError(agg)
    return pb


def continuous_covariate_de(pseudobulk: pd.DataFrame, covariate: pd.Series,
                             pseudocount: float = 1e-4) -> pd.DataFrame:
    """Per-gene Spearman (rank) correlation of pseudobulk expression with a
    continuous covariate (e.g. donor-level CPS score) - rank-based to stay in
    the same non-parametric spirit as wilcoxon_two_group_de, and more robust
    to the handful-of-donors sample sizes here. 'log2FC' is repurposed as the
    linear-fit slope (log2-expression per unit covariate) so the schema stays
    uniform; pvalue/padj come from the correlation test itself.
    """
    covariate = covariate.reindex(pseudobulk.index)
    valid = covariate.notna()
    x = covariate[valid].values.astype(float)
    genes = pseudobulk.columns
    rows = []
    logexpr = np.log2(pseudobulk.loc[valid] + pseudocount)
    for g in genes:
        y = logexpr[g].values
        if np.std(x) == 0 or np.std(y) == 0:
            r, p, slope = np.nan, np.nan, np.nan
        else:
            r, p = stats.spearmanr(x, y)
            slope = np.polyfit(x, y, 1)[0]
        rows.append((g, slope, p, r))
    out = pd.DataFrame(rows, columns=["gene", "log2FC", "pvalue", "spearman_r"])
    out["padj"] = _bh(out["pvalue"].values)
    out["n_group1"] = int(valid.sum())
    out["n_group2"] = None
    return out[["gene", "log2FC", "pvalue", "padj", "n_group1", "n_group2"]]
