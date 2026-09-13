"""
Paper 3 - CosMx Amyloid Plaque Niche (self-computed, CONFIRMED - see manifest
notes: no author gene-level DE table is deposited, only a pathway-level GSEA
table in the bioRxiv preprint's Supplementary Table 1).

Data on disk: 03_CosMx_AmyloidPlaqueNiche/GSE263791_RAW.tar, 2 CosMx samples
(GSM8199188, GSM8199189), each with a per-cell exprMat_file.csv.gz (950-gene
panel + 10 NegPrb controls) and a metadata_file.csv.gz containing per-cell
immunofluorescence intensities, including Mean.BetaAmyloid - a direct per-cell
proxy for plaque proximity (higher = closer to/within amyloid pathology),
matching the paper's own plaque-niche design (Comparison Type: "Plaque
proximity; disease timepoint").

We bin cells into plaque-proximal (top quartile Mean.BetaAmyloid) vs
plaque-distal (bottom quartile) within each sample.

REWRITTEN 2026-08-04 to use the project's standardized primary DE method (see
00_PROJECT_NOTES_AND_METHODOLOGY.txt section 2): per-CELL Wilcoxon rank-sum on
normalize_cells() output, comparing proximal vs distal cells directly, instead
of the earlier per-(sample,FOV)-pseudobulk Welch t-test.

No cell-type stratification here (exprMat file has no cell-type calls) -
cell_type is reported as "all". A future enhancement could add
Leiden/marker-based cell typing, matching the paper's own microglia/astrocyte-
specific analysis more closely.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import tarfile, io

import numpy as np
import pandas as pd
from de_pipeline import datasets, schema, stats_utils
from de_pipeline.paths import DATA_ROOT, DE_TABLES_DIR


TAR = str(DATA_ROOT / "03_CosMx_AmyloidPlaqueNiche/GSE263791_RAW.tar")
PANEL_CSV = str(DATA_ROOT / "03_CosMx_AmyloidPlaqueNiche/panel_list.csv")
PAPER_ID = "03_CosMx_AmyloidPlaqueNiche"

SAMPLES = ["GSM8199188_ID61-ID62_S10", "GSM8199189_ID67-ID68_S18"]


def _read_member(tf, name):
    return pd.read_csv(io.BytesIO(tf.extractfile(name).read()), compression="gzip")


def load_sample(tf, sample_prefix):
    expr = _read_member(tf, f"{sample_prefix}_exprMat_file.csv.gz")
    meta = _read_member(tf, f"{sample_prefix}_metadata_file.csv.gz")
    df = expr.merge(meta[["fov", "cell_ID", "Mean.BetaAmyloid"]], on=["fov", "cell_ID"], how="inner")
    return df


def main():
    panel = pd.read_csv(PANEL_CSV)
    target_genes = panel.loc[panel["category"] == "target", "gene_symbol"].tolist()

    with tarfile.open(TAR, "r") as tf:
        dfs = []
        for s in SAMPLES:
            print(f"[{PAPER_ID}] loading {s} ...")
            d = load_sample(tf, s)
            d["sample"] = s
            dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    print(f"  total cells: {len(df):,}")

    gene_cols = [g for g in target_genes if g in df.columns]
    print(f"  using {len(gene_cols)}/{len(target_genes)} target genes present in exprMat columns")

    # proximal/distal bin per sample (quartiles computed within-sample to control for staining batch effects)
    df["bin"] = None
    for s in SAMPLES:
        m = df["sample"] == s
        q75, q25 = df.loc[m, "Mean.BetaAmyloid"].quantile([0.75, 0.25])
        df.loc[m & (df["Mean.BetaAmyloid"] >= q75), "bin"] = "proximal"
        df.loc[m & (df["Mean.BetaAmyloid"] <= q25), "bin"] = "distal"

    binned = df[df["bin"].notna()].copy()
    raw = binned[gene_cols].values.astype(np.float32)
    norm = stats_utils.normalize_cells(raw)
    labels = binned["bin"].values
    print(f"  binned cells: {len(binned):,} (proximal={sum(labels=='proximal'):,}, "
          f"distal={sum(labels=='distal'):,})")

    de = stats_utils.wilcoxon_two_group_de(norm, gene_cols, labels, "proximal", "distal")
    de["cell_type"] = "all"
    de["comparison_type"] = "plaque-proximal vs plaque-distal (Mean.BetaAmyloid quartiles)"
    de["list_id"] = f"{PAPER_ID}__proximal_vs_distal__all"
    datasets.stamp(de, PAPER_ID)

    out = schema.make_df(de.to_dict("records"))
    outpath = str(DE_TABLES_DIR / "03_CosMx_AmyloidPlaqueNiche_DE.csv")
    schema.write_table(out, outpath)
    return out


if __name__ == "__main__":
    main()
