"""
Section 3.8c follow-up: pseudobulk (per-sample) vs per-cell Wilcoxon DE
concordance for dataset 07b (MERFISH PU.1 / Lymphoid Microglia, FADPU vs
FADTV), 8 samples (5 FADPU, 3 FADTV). Single panel-wide list, no cell-type
stratification (matches this dataset's own primary DE design).

Adapted from run_paper07b_selfcomputed.py's loading code, keeping per-sample
identity through to a pseudobulk aggregation instead of pooling straight to
condition.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import gc, tarfile, io

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from de_pipeline import stats_utils, enrichment as en
from de_pipeline.paths import DATA_ROOT, DE_TABLES_DIR
from pseudobulk_de import pseudobulk_two_group_de

TAR = str(DATA_ROOT / "07b_MERFISH_PU1_LymphoidMicroglia/GSE275026_RAW.tar")
PAPER_ID = "07b_MERFISH_PU1_LymphoidMicroglia"

SAMPLES = {
    "GSM8464544_counts_and_metadata_FADTV_635": "FADTV",
    "GSM8464545_counts_and_metadata_FADTV_348": "FADTV",
    "GSM8464548_counts_and_metadata_FADTV_411": "FADTV",
    "GSM8464546_counts_and_metadata_FADPU_538": "FADPU",
    "GSM8464547_counts_and_metadata_FADPU_521": "FADPU",
    "GSM8970742_counts_and_metadata_FADPU_663": "FADPU",
    "GSM8970743_counts_and_metadata_FADPU_662": "FADPU",
    "GSM8970744_counts_and_metadata_FADPU_668": "FADPU",
}
META_COLS = ["fov", "volume", "center_x", "center_y", "min_x", "min_y", "max_x", "max_y",
             "anisotropy", "transcript_count", "perimeter_area_ratio", "solidity"]
MAX_CELLS_PER_SAMPLE = 30_000
RNG_SEED = 0


def main():
    gene_cols = None
    blocks, conditions, sample_ids = [], [], []
    rng = np.random.default_rng(RNG_SEED)
    with tarfile.open(TAR, "r") as tf:
        for prefix, cond in SAMPLES.items():
            member = f"{prefix}.csv.gz"
            print(f"[{PAPER_ID}] extracting {prefix} ({cond}) ...")
            chunk = pd.read_csv(io.BytesIO(tf.extractfile(member).read()), compression="gzip", index_col=0)
            if gene_cols is None:
                gene_cols = [c for c in chunk.columns if c not in META_COLS and not c.startswith("Blank")]
            arr = chunk[gene_cols].values.astype(np.float32)
            if arr.shape[0] > MAX_CELLS_PER_SAMPLE:
                keep = rng.choice(arr.shape[0], size=MAX_CELLS_PER_SAMPLE, replace=False)
                arr = arr[keep]
            blocks.append(arr)
            conditions.extend([cond] * arr.shape[0])
            sample_ids.extend([prefix] * arr.shape[0])
            print(f"  kept {arr.shape[0]:,} cells")
            del chunk
            gc.collect()

    raw = np.vstack(blocks).astype(np.float32)
    del blocks
    gc.collect()
    conditions = np.asarray(conditions)
    sample_ids = np.asarray(sample_ids)
    print(f"  total: {raw.shape[0]:,} cells x {raw.shape[1]} genes "
          f"(FADPU={sum(conditions=='FADPU'):,}, FADTV={sum(conditions=='FADTV'):,})")

    norm = stats_utils.normalize_cells(raw)

    de_cell = stats_utils.wilcoxon_two_group_de(norm, gene_cols, conditions, "FADPU", "FADTV")
    sig_cell = set(de_cell.loc[de_cell["padj"] < 0.05, "gene"])
    print(f"  per-cell: {len(sig_cell)}/{len(gene_cols)} significant")

    de_pb = pseudobulk_two_group_de(norm, gene_cols, conditions, sample_ids,
                                     "FADPU", "FADTV", dict(SAMPLES))
    sig_pb = set(de_pb.loc[de_pb["padj"] < 0.05, "gene"])
    n_testable = int(de_pb["padj"].notna().sum())
    print(f"  pseudobulk (5 vs 3 samples): {len(sig_pb)}/{n_testable} testable significant")

    de_cell.to_csv(DE_TABLES_DIR / "paper07b_pseudobulk_check_percell.csv", index=False)
    de_pb.to_csv(DE_TABLES_DIR / "paper07b_pseudobulk_check_pseudobulk.csv", index=False)

    merged = de_cell.merge(de_pb, on="gene", suffixes=("_cell", "_pb"))
    rho, rho_p = spearmanr(merged["pvalue_cell"], merged["pvalue_pb"])
    print(f"  Spearman rho(p_cell, p_pseudobulk) = {rho:.3f} (p={rho_p:.2e})")

    panel = list(gene_cols)
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
