"""
Paper 5 - Xenium Parvalbumin Interneuron / Retrosplenial Cortex (self-computed
- CONFIRMED, see manifest: the paper's own supplementary tables S1-S3 contain
only an antibody list, a brain-region abbreviation key, and behavioral/
histology figure statistics - no gene-level MERFISH/Xenium DE table at all).

Data on disk: 05_Xenium_PVInterneuron_RetrosplenialCortex/GSE277463_RAW.tar,
6 samples in standard 10x (Xenium) mtx format: WT1F/WT2F/WT3F (wild-type,
female) vs TG2F/TG3F/TG4F (5xFAD transgenic, female). Xenium panel: 541 features in features.tsv, of which 247 are real genes
("Gene Expression") and 294 are control probes - see read_features().

REWRITTEN 2026-08-04 to use the project's standardized primary DE method (see
00_PROJECT_NOTES_AND_METHODOLOGY.txt section 2): per-CELL Wilcoxon rank-sum
on normalize_cells() output, pooling all cells within each genotype across its
3 mice, rather than the per-mouse pseudobulk Welch t-test used previously.
Not yet cell-type resolved - the paper's own cell typing lives in the .RDS
Seurat objects, which would need R/rpy2 to parse; documented as a follow-up.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import gc, tarfile, io, gzip

import numpy as np
import pandas as pd
from scipy.io import mmread
from de_pipeline import datasets, schema, stats_utils
from de_pipeline.paths import DATA_ROOT, DE_TABLES_DIR


TAR = str(DATA_ROOT / "05_Xenium_PVInterneuron_RetrosplenialCortex/GSE277463_RAW.tar")
PAPER_ID = "05_Xenium_PVInterneuron_RetrosplenialCortex"

SAMPLES = {
    "GSM8522749_WT1F": "WT", "GSM8522750_WT2F": "WT", "GSM8522751_WT3F": "WT",
    "GSM8522752_TG2F": "5xFAD", "GSM8522753_TG3F": "5xFAD", "GSM8522754_TG4F": "5xFAD",
}


# Xenium features.tsv lists real genes AND control probes in one file, tagged in
# its third column. Only "Gene Expression" rows are real measured genes; the rest
# are QC/control features that must never enter a DE test or a background gene set.
# For this dataset that is 247 genes out of 541 total rows (27 Negative Control
# Probe, 41 Negative Control Codeword, 225 Unassigned Codeword, 1 Deprecated
# Codeword). Equivalent to run_paper07b_selfcomputed.py's `not c.startswith("Blank")`
# filter and paper 03's negative-control exclusion - see notes Section 22 for the
# bug this fixes (all 541 rows were previously treated as genes).
GENE_FEATURE_KIND = "Gene Expression"


def read_features(tf, prefix):
    """Returns (all_symbols, keep_mask) - the mask selects real genes only.
    The mask is needed as well as the names because the count matrix has one
    row per features.tsv line, so columns must be subset by position."""
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
        for prefix, genotype in SAMPLES.items():
            feats, mask = read_features(tf, prefix)
            if feats is not None:
                gene_names, gene_mask = feats, mask
            assert gene_names is not None, f"no features.tsv found yet for {prefix}"
            raw = tf.extractfile(f"{prefix}_matrix.mtx.gz").read()
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
                mat = mmread(gz).tocsr()  # features x cells, raw counts
            dense = np.asarray(mat.T.todense(), dtype=np.float32)  # cells x features
            assert dense.shape[1] == len(gene_mask), (
                f"{prefix}: matrix has {dense.shape[1]} features but features.tsv has "
                f"{len(gene_mask)} rows - cannot align the gene filter")
            dense = dense[:, gene_mask]  # drop control probes BEFORE normalization
            blocks.append(dense)
            genotypes.extend([genotype] * dense.shape[0])
            print(f"[{PAPER_ID}] {prefix} ({genotype}): {dense.shape[0]:,} cells, {dense.shape[1]} genes")
            del mat, dense
            gc.collect()

    raw_expr = np.vstack(blocks).astype(np.float32)
    del blocks
    gc.collect()
    genotypes = np.asarray(genotypes)
    print(f"  total: {raw_expr.shape[0]:,} cells x {raw_expr.shape[1]} genes")

    norm = stats_utils.normalize_cells(raw_expr)  # normalize_cells operates in-place on raw_expr
    print("  normalized")
    gc.collect()

    de = stats_utils.wilcoxon_two_group_de(norm, gene_names, genotypes, "5xFAD", "WT")
    print("  wilcoxon done")
    de["cell_type"] = "all"
    de["comparison_type"] = "5xFAD vs WT"
    de["list_id"] = f"{PAPER_ID}__5xFAD_vs_WT__all"
    datasets.stamp(de, PAPER_ID)

    out = schema.make_df(de.to_dict("records"))
    outpath = str(DE_TABLES_DIR / "05_Xenium_PVInterneuron_DE.csv")
    schema.write_table(out, outpath)
    return out


if __name__ == "__main__":
    main()
