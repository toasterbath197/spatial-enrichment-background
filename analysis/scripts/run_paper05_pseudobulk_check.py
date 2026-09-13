"""
Fix #1 / Section 3.8c diagnostic: pseudobulk (sample-level) vs per-cell
Wilcoxon DE concordance for dataset 05 (Xenium, PV interneuron / retrosplenial
cortex, 5xFAD vs WT, 3 mice per genotype). Chosen for this check because it
has unambiguous biological-replicate structure (6 named mice, 3 per group)
and is small enough to run in a memory-constrained environment.

NOT run for datasets 01 or 03 (together 46 of 54 baseline lists) - both
exceed the storage/memory budget this check was developed under. See
00_PROJECT_NOTES_AND_METHODOLOGY.txt section 41 for status and follow-up.

Adapted from run_paper05_xenium.py (same loading code), with sample identity
retained instead of collapsed to genotype only, so cells can be aggregated to
one pseudobulk row per mouse before testing. Writes two DE tables (per-cell,
pseudobulk) to results/DE_tables/ for inspection; does not touch
master_DE_table.csv or any file the primary pipeline reads.

    python3 run_paper05_pseudobulk_check.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import gc, tarfile, io, gzip

import numpy as np
import pandas as pd
from scipy.io import mmread
from de_pipeline import stats_utils, enrichment as en
from de_pipeline.paths import DATA_ROOT, DE_TABLES_DIR
from pseudobulk_de import pseudobulk_two_group_de

TAR = str(DATA_ROOT / "05_Xenium_PVInterneuron_RetrosplenialCortex/GSE277463_RAW.tar")
PAPER_ID = "05_Xenium_PVInterneuron_RetrosplenialCortex"
GENE_FEATURE_KIND = "Gene Expression"

SAMPLES = {
    "GSM8522749_WT1F": "WT", "GSM8522750_WT2F": "WT", "GSM8522751_WT3F": "WT",
    "GSM8522752_TG2F": "5xFAD", "GSM8522753_TG3F": "5xFAD", "GSM8522754_TG4F": "5xFAD",
}


def read_features(tf, prefix):
    member = f"{prefix}_features.tsv.gz"
    try:
        raw = tf.extractfile(member).read()
    except KeyError:
        return None, None
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
        df = pd.read_csv(gz, sep="\t", header=None, names=["ensembl", "symbol", "kind"])
    keep = (df["kind"] == GENE_FEATURE_KIND).values
    return df.loc[keep, "symbol"].tolist(), keep


def main():
    with tarfile.open(TAR, "r") as tf:
        gene_names = None
        gene_mask = None
        blocks = []
        genotypes = []
        sample_ids = []
        for prefix, genotype in SAMPLES.items():
            feats, mask = read_features(tf, prefix)
            if feats is not None:
                gene_names, gene_mask = feats, mask
            raw = tf.extractfile(f"{prefix}_matrix.mtx.gz").read()
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
                mat = mmread(gz).tocsr()
            dense = np.asarray(mat.T.todense(), dtype=np.float32)
            dense = dense[:, gene_mask]
            blocks.append(dense)
            genotypes.extend([genotype] * dense.shape[0])
            sample_ids.extend([prefix] * dense.shape[0])
            print(f"[{PAPER_ID}] {prefix} ({genotype}): {dense.shape[0]:,} cells")
            del mat, dense
            gc.collect()

    raw_expr = np.vstack(blocks).astype(np.float32)
    del blocks
    gc.collect()
    genotypes = np.asarray(genotypes)
    sample_ids = np.asarray(sample_ids)
    print(f"  total: {raw_expr.shape[0]:,} cells x {raw_expr.shape[1]} genes")

    norm = stats_utils.normalize_cells(raw_expr)
    print("  normalized")

    # --- Tier A: per-cell Wilcoxon (the project's primary method, reproduced) ---
    de_cell = stats_utils.wilcoxon_two_group_de(norm, gene_names, genotypes, "5xFAD", "WT")
    print(f"  per-cell: {int((de_cell['padj'] < 0.05).sum())} genes padj<0.05 of {len(de_cell)}")

    # --- Tier B: pseudobulk (one row per mouse, 3 vs 3) ---
    de_pb = pseudobulk_two_group_de(norm, gene_names, genotypes, sample_ids,
                                     "5xFAD", "WT", dict(SAMPLES))
    n_pb_sig = int((de_pb["padj"] < 0.05).sum())
    n_pb_testable = int(de_pb["padj"].notna().sum())
    print(f"  pseudobulk: {n_pb_sig} genes padj<0.05 of {n_pb_testable} testable "
          f"({len(de_pb)} total, 3 vs 3 mice)")

    de_cell.to_csv(DE_TABLES_DIR / "paper05_pseudobulk_check_percell.csv", index=False)
    de_pb.to_csv(DE_TABLES_DIR / "paper05_pseudobulk_check_pseudobulk.csv", index=False)

    # --- Concordance ---
    sig_cell = set(de_cell.loc[de_cell["padj"] < 0.05, "gene"])
    sig_pb = set(de_pb.loc[de_pb["padj"] < 0.05, "gene"])
    print(f"\n  per-cell significant: {len(sig_cell)}")
    print(f"  pseudobulk significant: {len(sig_pb)}")
    print(f"  overlap: {len(sig_cell & sig_pb)}")

    merged = de_cell.merge(de_pb, on="gene", suffixes=("_cell", "_pb"))
    from scipy.stats import spearmanr
    rho, rho_p = spearmanr(merged["pvalue_cell"], merged["pvalue_pb"])
    print(f"  Spearman rho(p-value_cell, p-value_pseudobulk) = {rho:.3f} (p={rho_p:.2e})")

    # --- Downstream: does the enrichment conclusion change? ---
    panel = list(gene_names)
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
