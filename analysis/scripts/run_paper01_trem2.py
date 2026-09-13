"""
Paper 1 - Trem2R47H MERFISH (self-computed primary tier).

Data on disk: 01_Trem2R47H_MERFISH/MERFISH_Data.h5ad, 432,794 cells x 300-gene
VZG171 panel. obs['gen_coarse'] = 4 genotype groups (WT/5xFAD/Trem2/
Trem2_5xFAD); obs['cluster_coarse'] = 9 major cell types.

REWRITTEN 2026-08-04 to use the project's standardized primary DE method (see
00_PROJECT_NOTES_AND_METHODOLOGY.txt section 2): per-CELL Wilcoxon rank-sum on
normalize_cells() output of the RAW counts layer ('RNA'), replacing the
earlier per-mouse pseudobulk Welch t-test. This is a self-computed
cross-check/fallback regardless - the author's own Supplementary Tables 3-5
(not yet downloaded) remain the preferred tier-2 comparison for this paper
once pulled (see manifest notes for paper 1).

Runs one cell type at a time (pass it as argv[1]) so each invocation finishes
comfortably inside this environment's per-command time/memory budget - 9 cell
types x 5 genotype comparisons is too much Wilcoxon-on-raw-cells work for one
call. Writes results/DE_tables/_parts/01_<celltype>.csv per cell type;
passing --all (or one cell type at a time, finishing with --finalize)
concatenates the parts into the paper's final DE table once all 9 exist.

Usage:
    python3 run_paper01_trem2.py "Mic"
    python3 run_paper01_trem2.py --all       # do it all in one go if you have time/memory budget
    python3 run_paper01_trem2.py --finalize  # just concatenate existing parts
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import gc

import numpy as np
import pandas as pd
from de_pipeline import datasets, h5ad_io, schema, stats_utils
from de_pipeline.paths import DATA_ROOT, DE_TABLES_DIR, PARTS_DIR


H5AD = str(DATA_ROOT / "01_Trem2R47H_MERFISH/MERFISH_Data.h5ad")
PAPER_ID = "01_Trem2R47H_MERFISH"
PARTS_DIR = PARTS_DIR

COMPARISONS = [
    ("5xFAD", "WT"),
    ("Trem2", "WT"),
    ("Trem2_5xFAD", "Trem2"),
    ("Trem2_5xFAD", "5xFAD"),
    ("Trem2_5xFAD", "WT"),
]

CELL_TYPES = ["AST", "Amygdala ExN", "Cortical ExN", "Cortical Inh", "Hippocampal ExN",
              "Mic", "OGC", "Other Glia", "Subcortical Neurons"]


def run_cell_type(ct: str):
    print(f"[{PAPER_ID}] loading obs columns + raw counts for cell type '{ct}' ...")
    genes = h5ad_io.read_var_names(H5AD)
    gen_coarse = h5ad_io.read_obs_column(H5AD, "gen_coarse").values
    cluster_coarse = h5ad_io.read_obs_column(H5AD, "cluster_coarse").values
    mask = cluster_coarse == ct
    if mask.sum() < 50:
        print(f"  [skip] too few cells for {ct}")
        return None

    import h5py
    with h5py.File(H5AD, "r") as f:
        idx = np.where(mask)[0]
        raw = f["layers/RNA"][:][idx, :].astype(np.float32)  # subset then cast down
    norm = stats_utils.normalize_cells(raw)
    labels = gen_coarse[mask]
    print(f"  {mask.sum():,} cells")

    rows = []
    for g1, g2 in COMPARISONS:
        try:
            de = stats_utils.wilcoxon_two_group_de(norm, genes, labels, g1, g2)
        except ValueError as e:
            print(f"  [skip] {g1} vs {g2}: {e}")
            continue
        de["cell_type"] = ct
        de["comparison_type"] = f"{g1} vs {g2}"
        de["list_id"] = f"{PAPER_ID}__{g1}_vs_{g2}__{ct}"
        rows.append(de)
        print(f"  done {g1} vs {g2}")

    if not rows:
        return None
    out = pd.concat(rows, ignore_index=True)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_ct = ct.replace(" ", "_")
    outpath = PARTS_DIR / f"01_{safe_ct}.csv"
    out.to_csv(outpath, index=False)
    print(f"  wrote {len(out):,} rows -> {outpath}")
    del raw, norm
    gc.collect()
    return out


def finalize():
    """Concatenate all per-cell-type parts into the paper's final DE table."""
    parts = sorted(PARTS_DIR.glob("01_*.csv"))
    have = {p.stem[3:].replace("_", " ") for p in parts}
    missing = set(CELL_TYPES) - have
    if missing:
        print(f"  NOT finalizing yet - missing cell types: {missing}")
        return None
    de_all = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    datasets.stamp(de_all, PAPER_ID)
    out = schema.make_df(de_all.to_dict("records"))
    outpath = DE_TABLES_DIR / "01_Trem2R47H_MERFISH_DE.csv"
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
