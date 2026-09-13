"""
Paper 7b - MERFISH PU.1 / Lymphoid Microglia - SELF-COMPUTED primary tier.

Per the two-tier design (00_PROJECT_NOTES_AND_METHODOLOGY.txt section 2), a
paper having a genuine author-published DE table (confirmed here -
Supplementary Table 4, see run_paper07b_author_table.py) makes that table a
SECONDARY robustness check, not a replacement for this paper also having a
self-computed PRIMARY entry like every other paper.

Data on disk: 07b_MERFISH_PU1_LymphoidMicroglia/GSE275026_RAW.tar, 8 raw
per-sample MERFISH counts-and-metadata CSVs (~950k cells total, 398-gene
panel + 64 Blank negative-control probes + per-cell spatial/QC metadata -
NO cell-type calls and NO plaque-distance column in this public deposit).

IMPORTANT DEVIATION FROM THE AUTHOR TABLE'S COMPARISON: the author table's
comparison is plaque-associated vs. distal MICROGLIA (a spatial, cell-type-
specific axis) - that requires plaque segmentation coordinates that are not
part of this GEO deposit, so we cannot faithfully self-compute that exact
comparison from what's public here. What IS fully derivable from the sample
names is the condition axis: FADTV (3 samples, control-virus) vs FADPU (5
samples, PU.1-dosage-manipulated) - the paper's genetic manipulation arm. So
the self-computed primary tier here is FADPU vs FADTV, panel-wide (no
cell-type stratification - not derivable from this deposit either).

MEMORY/TIME NOTE: ~950k cells doesn't fit this environment's ~3.8GB RAM /
45s-per-command budget in one shot, whether for CSV parsing+decompression or
for the Wilcoxon test itself. Split into two phases, run across separate
commands:
    python3 run_paper07b_selfcomputed.py extract <prefix>   # one of the 8 samples, x8 calls
    python3 run_paper07b_selfcomputed.py finalize           # combine + normalize + Wilcoxon
Each sample is subsampled to MAX_CELLS_PER_SAMPLE cells (fixed seed,
reproducible) during extraction - standard practice at this scale (e.g.
Seurat's max.cells.per.ident), and still leaves each condition with tens of
thousands of cells.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import gc, tarfile, io

import numpy as np
import pandas as pd
from de_pipeline import datasets, schema, stats_utils
from de_pipeline.paths import DATA_ROOT, DE_TABLES_DIR, PARTS_DIR


TAR = str(DATA_ROOT / "07b_MERFISH_PU1_LymphoidMicroglia/GSE275026_RAW.tar")
PAPER_ID = "07b_MERFISH_PU1_LymphoidMicroglia"
PARTS_DIR = PARTS_DIR / "07b_raw"

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


def extract(prefix: str):
    cond = SAMPLES[prefix]
    member = f"{prefix}.csv.gz"
    print(f"[{PAPER_ID}] extracting {prefix} ({cond}) ...")
    with tarfile.open(TAR, "r") as tf:
        chunk = pd.read_csv(io.BytesIO(tf.extractfile(member).read()), compression="gzip", index_col=0)
    gene_cols = [c for c in chunk.columns if c not in META_COLS and not c.startswith("Blank")]
    arr = chunk[gene_cols].values.astype(np.float32)
    rng = np.random.default_rng(RNG_SEED)
    if arr.shape[0] > MAX_CELLS_PER_SAMPLE:
        keep = rng.choice(arr.shape[0], size=MAX_CELLS_PER_SAMPLE, replace=False)
        arr = arr[keep]
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PARTS_DIR / f"{prefix}.npy", arr)
    (PARTS_DIR / f"{prefix}.condition.txt").write_text(cond)
    if not (PARTS_DIR / "gene_cols.txt").exists():
        (PARTS_DIR / "gene_cols.txt").write_text("\n".join(gene_cols))
    print(f"  saved {arr.shape[0]:,} cells x {arr.shape[1]} genes")


def finalize():
    gene_cols = (PARTS_DIR / "gene_cols.txt").read_text().splitlines()
    blocks, conditions = [], []
    for prefix, cond in SAMPLES.items():
        p = PARTS_DIR / f"{prefix}.npy"
        if not p.exists():
            print(f"  NOT finalizing yet - missing extract for {prefix}")
            return None
        arr = np.load(p)
        blocks.append(arr)
        conditions.extend([cond] * arr.shape[0])
    raw = np.vstack(blocks).astype(np.float32)
    del blocks
    gc.collect()
    conditions = np.asarray(conditions)
    print(f"  total: {raw.shape[0]:,} cells x {raw.shape[1]} genes "
          f"(FADPU={sum(conditions=='FADPU'):,}, FADTV={sum(conditions=='FADTV'):,})")

    norm = stats_utils.normalize_cells(raw)
    de = stats_utils.wilcoxon_two_group_de(norm, gene_cols, conditions, "FADPU", "FADTV")
    de["cell_type"] = "all"
    de["comparison_type"] = "FADPU (PU.1-dosage-manipulated) vs FADTV (control-virus), panel-wide"
    de["list_id"] = f"{PAPER_ID}__FADPU_vs_FADTV__all"
    datasets.stamp(de, PAPER_ID)

    out = schema.make_df(de.to_dict("records"))
    outpath = str(DE_TABLES_DIR / "07b_MERFISH_PU1_selfcomputed_DE.csv")
    schema.write_table(out, outpath)
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] == "extract":
        extract(sys.argv[2])
    elif sys.argv[1] == "finalize":
        finalize()
    else:
        print(__doc__)
