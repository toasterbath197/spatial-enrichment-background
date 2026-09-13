"""
Paper 6 - Xenium APOE4 vs APOE3 (Millet/Tavazoie, Immunity 2024) - self-computed
primary tier.

UNBLOCKED 2026-08-04: merged_xenium.h5ad (314.6MB, pre-merged analysis-ready
object across 6 donors: 3x APOE3, 3x APOE4) downloaded from Zenodo
10.5281/zenodo.8206638 and placed in 06_Xenium_APOE4vAPOE3/ by the user - see
that folder's README.txt and 00_PROJECT_NOTES_AND_METHODOLOGY.txt section 1
for why only this one file (not the ~76GB of raw per-donor Xenium output or
~12GB of methoxy-x04 whole-slide scans) was needed.

Data: 494,376 cells x 266 genes. obs['genotype'] = E3/E4 (APOE3/APOE4).
obs['sample'] = 6 donors (E3_1..3, E4_1..3). obs['superclust'] = 5 broad cell
types (Neurons, Oligodendrocytes, Astrocytes, Microglia, Other) - 'Microglia'
maps directly onto the paper's own focus (the TIM microglial subpopulation).
X is already log1p-normalized by the authors; layers/counts holds raw counts -
we use layers/counts and apply this project's own normalize_cells() to it, per
the standardized-method rule (00_PROJECT_NOTES_AND_METHODOLOGY.txt section 2):
per-CELL Wilcoxon rank-sum on normalize_cells() output, APOE4 vs APOE3, per
cell type - same method as papers 1, 3, 5.

Runs one cell type at a time (pass as argv[1]) for the same
memory/time-budget reasons as run_paper01_trem2.py; writes
results/DE_tables/_parts/06_<celltype>.csv per cell type and finalizes once
all 5 exist.

Usage:
    python3 run_paper06_apoe.py "Microglia"
    python3 run_paper06_apoe.py --all
    python3 run_paper06_apoe.py --finalize
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import gc

import numpy as np
import pandas as pd
import h5py
from scipy.sparse import csr_matrix
from de_pipeline import datasets, h5ad_io, schema, stats_utils
from de_pipeline.paths import DATA_ROOT, DE_TABLES_DIR, PARTS_DIR


H5AD = str(DATA_ROOT / "06_Xenium_APOE4vAPOE3/merged_xenium.h5ad")
PAPER_ID = "06_Xenium_APOE4vAPOE3"
PARTS_DIR = PARTS_DIR

CELL_TYPES = ["Neurons", "Oligodendrocytes", "Astrocytes", "Microglia", "Other"]


def run_cell_type(ct: str):
    print(f"[{PAPER_ID}] loading obs columns + raw counts for cell type '{ct}' ...")
    genes = h5ad_io.read_var_names(H5AD)
    genotype = h5ad_io.read_obs_column(H5AD, "genotype").values
    superclust = h5ad_io.read_obs_column(H5AD, "superclust").values
    mask = superclust == ct
    if mask.sum() < 50:
        print(f"  [skip] too few cells for {ct}")
        return None

    with h5py.File(H5AD, "r") as f:
        grp = f["layers/counts"]
        n_obs, n_var = grp.attrs["shape"]
        full = csr_matrix((grp["data"][:], grp["indices"][:], grp["indptr"][:]), shape=(n_obs, n_var))
        sub = full[mask, :]  # sparse row slicing - avoids densifying the full 494k-cell matrix
        raw = np.asarray(sub.todense(), dtype=np.float32)
        del full, sub

    norm = stats_utils.normalize_cells(raw)
    labels = genotype[mask]
    print(f"  {mask.sum():,} cells (E3={sum(labels=='E3'):,}, E4={sum(labels=='E4'):,})")

    try:
        de = stats_utils.wilcoxon_two_group_de(norm, genes, labels, "E4", "E3")
    except ValueError as e:
        print(f"  [skip] E4 vs E3: {e}")
        return None
    de["cell_type"] = ct
    de["comparison_type"] = "APOE4 vs APOE3"
    de["list_id"] = f"{PAPER_ID}__APOE4_vs_APOE3__{ct}"
    print(f"  done E4 vs E3 for {ct}")

    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    outpath = PARTS_DIR / f"06_{ct}.csv"
    de.to_csv(outpath, index=False)
    print(f"  wrote {len(de):,} rows -> {outpath}")
    del raw, norm
    gc.collect()
    return de


def finalize():
    parts = sorted(PARTS_DIR.glob("06_*.csv"))
    have = {p.stem[3:] for p in parts}
    missing = set(CELL_TYPES) - have
    if missing:
        print(f"  NOT finalizing yet - missing cell types: {missing}")
        return None
    de_all = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    datasets.stamp(de_all, PAPER_ID)
    out = schema.make_df(de_all.to_dict("records"))
    outpath = DE_TABLES_DIR / "06_Xenium_APOE4vAPOE3_DE.csv"
    schema.write_table(out, str(outpath))
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    arg = sys.argv[1]
    if arg == "--all":
        for ct in CELL_TYPES:
            run_cell_type(ct)
        finalize()
    elif arg == "--finalize":
        finalize()
    else:
        run_cell_type(arg)
        finalize()
