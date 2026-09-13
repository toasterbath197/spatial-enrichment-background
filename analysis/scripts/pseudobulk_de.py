"""
Reconstructed pseudobulk two-group DE test, for the pseudoreplication robustness
check only (Fix #1). Not added to stats_utils.py as project infrastructure -
this is a one-off diagnostic, kept separate and clearly labeled as such, per the
project's own convention of not leaving analysis-shaped code in the library
that only one script calls (see stats_utils.py module docstring on why the
original pseudobulk Welch t-test was removed on 2026-08-04).

Method: aggregate cells to one row per sample (mean of normalize_cells()
output, i.e. mean log-normalized expression per sample - same input the
per-cell Wilcoxon test uses, just aggregated first), then Mann-Whitney U
across samples between the two groups. With 3 samples per group this is the
same family of test as the primary method (rank-based, BH-corrected), just
applied at the sample level instead of the cell level, so the comparison
isolates the effect of the unit of replication rather than also changing the
test family.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def pseudobulk_two_group_de(norm_expr, gene_names, cell_group, sample_ids,
                             group1, group2, sample_to_group):
    """norm_expr: cells x genes, already normalize_cells() output.
    sample_ids: per-cell sample label. sample_to_group: {sample_id: group}.
    Returns DataFrame(gene, pvalue, padj, n_samples_group1, n_samples_group2).
    """
    df = pd.DataFrame(norm_expr, columns=gene_names)
    df["__sample__"] = sample_ids
    pb = df.groupby("__sample__").mean()  # samples x genes, mean log-normalized expr

    pb_groups = pb.index.map(sample_to_group)
    x1 = pb[pb_groups == group1].values
    x2 = pb[pb_groups == group2].values
    n1, n2 = x1.shape[0], x2.shape[0]

    n_genes = x1.shape[1]
    pvals = np.empty(n_genes, dtype=float)
    for g in range(n_genes):
        try:
            _, p = stats.mannwhitneyu(x1[:, g], x2[:, g], alternative="two-sided")
        except ValueError:
            p = np.nan  # identical values in both groups (all-zero gene, etc.)
        pvals[g] = p
    mask = ~np.isnan(pvals)
    padj = np.full(n_genes, np.nan)
    if mask.sum():
        padj[mask] = multipletests(pvals[mask], method="fdr_bh")[1]

    mean1 = np.expm1(x1).mean(axis=0)
    mean2 = np.expm1(x2).mean(axis=0)
    log2fc = np.log2(mean1 + 1e-9) - np.log2(mean2 + 1e-9)

    return pd.DataFrame({
        "gene": gene_names, "log2FC": log2fc, "pvalue": pvals, "padj": padj,
        "n_samples_group1": n1, "n_samples_group2": n2,
    })
