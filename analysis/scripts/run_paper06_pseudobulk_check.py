"""
Section 3.8c follow-up: pseudobulk (per-donor) vs per-cell Wilcoxon DE
concordance for dataset 06 (Xenium APOE4 vs APOE3, Millet/Tavazoie), run per
cell type to match how this dataset's baseline lists are scored (5 lists:
Neurons, Oligodendrocytes, Astrocytes, Microglia, Other). 6 donors, 3 per
genotype (E3_1..3, E4_1..3) - same replicate structure as dataset 05.

Adapted from run_paper06_apoe.py's loading code, keeping obs['sample'] (donor
ID) instead of collapsing straight to genotype.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import gc

import numpy as np
import pandas as pd
import h5py
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
from de_pipeline import h5ad_io, stats_utils, enrichment as en
from de_pipeline.paths import DATA_ROOT, DE_TABLES_DIR
from pseudobulk_de import pseudobulk_two_group_de

H5AD = str(DATA_ROOT / "06_Xenium_APOE4vAPOE3/merged_xenium.h5ad")
PAPER_ID = "06_Xenium_APOE4vAPOE3"
CELL_TYPES = ["Neurons", "Oligodendrocytes", "Astrocytes", "Microglia", "Other"]

SAMPLE_TO_GROUP = {
    "E3_1": "E3", "E3_2": "E3", "E3_3": "E3",
    "E4_1": "E4", "E4_2": "E4", "E4_3": "E4",
}


def main():
    print(f"[{PAPER_ID}] loading obs columns ...")
    genes = h5ad_io.read_var_names(H5AD)
    genotype = h5ad_io.read_obs_column(H5AD, "genotype").values
    superclust = h5ad_io.read_obs_column(H5AD, "superclust").values
    sample = h5ad_io.read_obs_column(H5AD, "sample").values
    print(f"  samples present: {sorted(set(sample))}")

    with h5py.File(H5AD, "r") as f:
        grp = f["layers/counts"]
        n_obs, n_var = grp.attrs["shape"]
        full = csr_matrix((grp["data"][:], grp["indices"][:], grp["indptr"][:]), shape=(n_obs, n_var))

    summary_rows = []
    gmt, universe = en.load_gmt("human"), en.gmt_universe("human")

    for ct in CELL_TYPES:
        mask = superclust == ct
        if mask.sum() < 50:
            print(f"  [skip] too few cells for {ct}")
            continue
        sub = full[mask, :]
        raw = np.asarray(sub.todense(), dtype=np.float32)
        norm = stats_utils.normalize_cells(raw)
        labels = genotype[mask]
        samp = sample[mask]
        print(f"\n  == {ct}: {mask.sum():,} cells (E3={sum(labels=='E3'):,}, E4={sum(labels=='E4'):,}) ==")

        de_cell = stats_utils.wilcoxon_two_group_de(norm, genes, labels, "E4", "E3")
        sig_cell = set(de_cell.loc[de_cell["padj"] < 0.05, "gene"])

        de_pb = pseudobulk_two_group_de(norm, genes, labels, samp, "E4", "E3", SAMPLE_TO_GROUP)
        sig_pb = set(de_pb.loc[de_pb["padj"] < 0.05, "gene"])

        merged = de_cell.merge(de_pb, on="gene", suffixes=("_cell", "_pb"))
        rho, rho_p = spearmanr(merged["pvalue_cell"], merged["pvalue_pb"])

        panel = list(genes)
        row = {"cell_type": ct, "n_cells": int(mask.sum()),
               "n_sig_percell": len(sig_cell), "n_sig_pseudobulk": len(sig_pb),
               "n_panel_genes": len(panel), "spearman_rho": rho, "spearman_p": rho_p}
        for label, query in (("percell", sig_cell), ("pseudobulk", sig_pb)):
            if len(query) < 3:
                row[f"panel_sig_{label}"], row[f"genome_sig_{label}"] = None, None
                continue
            panel_res = en.hypergeometric_enrichment(query, panel, gmt)
            genome_res = en.hypergeometric_enrichment(query, universe, gmt)
            row[f"panel_sig_{label}"] = int((panel_res["padj"] < 0.05).sum()) if len(panel_res) else 0
            row[f"genome_sig_{label}"] = int((genome_res["padj"] < 0.05).sum()) if len(genome_res) else 0
        summary_rows.append(row)
        print(f"  per-cell sig={len(sig_cell)}/{len(panel)}  pseudobulk sig={len(sig_pb)}/{len(panel)}  rho={rho:.3f}")
        del raw, norm, sub
        gc.collect()

    out = pd.DataFrame(summary_rows)
    out.to_csv(DE_TABLES_DIR / "paper06_pseudobulk_check_summary.csv", index=False)
    print("\n", out.to_string(index=False))


if __name__ == "__main__":
    main()
