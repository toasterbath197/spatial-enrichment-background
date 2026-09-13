"""
Per-gene detection rates, one reader per dataset.

A gene is "detected" if it has a non-zero count in at least DEFAULT_THRESHOLD of
cells. This is needed for the detected-genes background, which is the
intermediate option current guidance recommends.

Each reader opens the same raw source its dataset's differential-expression
driver opens, and applies the same control-feature exclusion, so the gene set
here always matches that dataset's panel. All of them stream in chunks or
sample-by-sample - none holds a full expression matrix in memory.

Used by: scripts/run_experiments.py detected-background
"""
import gc
import io
import gzip
import tarfile

import numpy as np
import pandas as pd
from scipy.io import mmread

from de_pipeline import h5ad_io

from de_pipeline.paths import DATA_ROOT, PARTS_DIR

DEFAULT_THRESHOLD = 0.01   # a gene counts as detected in >=1% of cells


def detection_rates_paper05():
    """Reparse the same GEO mtx tar run_paper05_xenium.py uses; return {gene: fraction_cells_detected}."""
    tar_path = DATA_ROOT / "05_Xenium_PVInterneuron_RetrosplenialCortex" / "GSE277463_RAW.tar"
    samples = ["GSM8522749_WT1F", "GSM8522750_WT2F", "GSM8522751_WT3F",
               "GSM8522752_TG2F", "GSM8522753_TG3F", "GSM8522754_TG4F"]
    gene_names = None
    n_cells_total = 0
    n_detected = None
    with tarfile.open(tar_path, "r") as tf:
        for prefix in samples:
            # not every sample has its own features.tsv.gz in this deposit (e.g. GSM8522751_WT3F
            # doesn't) - same fallback run_paper05_xenium.py's read_features() uses: reuse the last
            # successfully-loaded gene list, since the panel is identical across samples.
            try:
                raw_feat = tf.extractfile(f"{prefix}_features.tsv.gz").read()
                with gzip.GzipFile(fileobj=io.BytesIO(raw_feat)) as gz:
                    feats = pd.read_csv(gz, sep="\t", header=None, names=["ensembl", "symbol", "kind"])
                # match run_paper05_xenium.py: only "Gene Expression" rows are real genes,
                # the other 294 features are control probes / codewords (notes Section 22)
                gene_mask = (feats["kind"] == "Gene Expression").values
                gene_names = feats.loc[gene_mask, "symbol"].tolist()
            except KeyError:
                pass
            assert gene_names is not None, f"no features.tsv found yet for {prefix}"
            if n_detected is None:
                n_detected = np.zeros(len(gene_names), dtype=np.int64)
            raw_mat = tf.extractfile(f"{prefix}_matrix.mtx.gz").read()
            with gzip.GzipFile(fileobj=io.BytesIO(raw_mat)) as gz:
                mat = mmread(gz).tocsr()  # features x cells
            n_detected += np.asarray((mat > 0).sum(axis=1)).ravel()[gene_mask]
            n_cells_total += mat.shape[1]
            del mat
            gc.collect()
    rates = n_detected / n_cells_total
    return dict(zip(gene_names, rates))


def _h5ad_detection(path, node_key, chunk=40000, gene_filter=None):
    """Stream an h5ad X/layer in row chunks, counting nonzero cells per gene.
    Handles both dense and CSR-sparse encodings, same as de_pipeline/h5ad_io.py.
    Returns {gene: fraction_of_cells_with_nonzero_count}."""
    import h5py
    from scipy import sparse
    with h5py.File(path, "r") as f:
        genes = h5ad_io.read_var_names(path)
        node = f[node_key]
        if isinstance(node, h5py.Group):  # CSR sparse
            shape = tuple(node.attrs["shape"])
            indptr = node["indptr"][:]
            n_cells, n_genes = shape
            n_detected = np.zeros(n_genes, dtype=np.int64)
            for start in range(0, n_cells, chunk):
                end = min(start + chunk, n_cells)
                lo, hi = int(indptr[start]), int(indptr[end])
                sub = sparse.csr_matrix(
                    (node["data"][lo:hi], node["indices"][lo:hi], indptr[start:end + 1] - indptr[start]),
                    shape=(end - start, n_genes))
                n_detected += np.asarray((sub > 0).sum(axis=0)).ravel()
                del sub
                gc.collect()
        else:  # dense
            n_cells, n_genes = node.shape
            n_detected = np.zeros(n_genes, dtype=np.int64)
            for start in range(0, n_cells, chunk):
                end = min(start + chunk, n_cells)
                block = node[start:end, :]
                n_detected += (block > 0).sum(axis=0).astype(np.int64)
                del block
                gc.collect()
    rates = n_detected / n_cells
    out = dict(zip(genes, rates))
    if gene_filter is not None:
        out = {g: v for g, v in out.items() if gene_filter(g)}
    return out


def detection_rates_paper01():
    """h5ad layers/RNA (raw counts), same source as run_paper01_trem2.py."""
    return _h5ad_detection(
        str(DATA_ROOT / "01_Trem2R47H_MERFISH" / "MERFISH_Data.h5ad"), "layers/RNA")


def detection_rates_paper04():
    """h5ad X (raw MERFISH spot counts), same source as run_paper04_seaad.py.
    Blank-* control probes excluded, matching that driver (notes Section 22)."""
    return _h5ad_detection(
        str(DATA_ROOT / "04_SEAAD_MERFISH" / "SEAAD_MTG_MERFISH.2024-12-11.h5ad"), "X",
        gene_filter=lambda g: not str(g).lower().startswith("blank"))


def detection_rates_paper06():
    """h5ad layers/counts (raw), same source as run_paper06_apoe.py - NOT X,
    which the original authors already log-normalized."""
    return _h5ad_detection(
        str(DATA_ROOT / "06_Xenium_APOE4vAPOE3" / "merged_xenium.h5ad"), "layers/counts")


def detection_rates_paper03():
    """CosMx exprMat CSVs from the GEO tar, same source as run_paper03_cosmx.py.
    Restricted to 'target' genes from the panel CSV, so the 10 NegPrb control
    probes never enter."""
    tar_path = DATA_ROOT / "03_CosMx_AmyloidPlaqueNiche" / "GSE263791_RAW.tar"
    panel_csv = DATA_ROOT / "03_CosMx_AmyloidPlaqueNiche" / "panel_list.csv"
    samples = ["GSM8199188_ID61-ID62_S10", "GSM8199189_ID67-ID68_S18"]
    target = None
    if panel_csv.exists():
        p = pd.read_csv(panel_csv)
        target = set(p.loc[p["category"] == "target", "gene_symbol"])
    n_detected, n_cells_total, gene_cols = None, 0, None
    with tarfile.open(tar_path, "r") as tf:
        for s in samples:
            expr = pd.read_csv(io.BytesIO(tf.extractfile(f"{s}_exprMat_file.csv.gz").read()),
                               compression="gzip")
            if gene_cols is None:
                cand = [c for c in expr.columns if c not in ("fov", "cell_ID")]
                gene_cols = [c for c in cand if (target is None or c in target)
                             and not c.startswith("NegPrb")]
                n_detected = np.zeros(len(gene_cols), dtype=np.int64)
            arr = expr[gene_cols].values
            n_detected += (arr > 0).sum(axis=0).astype(np.int64)
            n_cells_total += arr.shape[0]
            del expr, arr
            gc.collect()
    return dict(zip(gene_cols, n_detected / n_cells_total))


def detection_rates_paper08():
    """Per-cell expression txt.gz, same source as run_paper08_tcellmyelin.py."""
    path = DATA_ROOT / "08_MERFISH_TcellMyelin" / "GSE243120_MERFISH_5xFAD_expression.txt.gz"
    header = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    gene_cols = [c for c in header if c != "cellID"]
    dtype_map = {c: "int32" for c in gene_cols}
    dtype_map["cellID"] = "int64"
    n_detected = np.zeros(len(gene_cols), dtype=np.int64)
    n_cells_total = 0
    for chunk_df in pd.read_csv(path, sep="\t", dtype=dtype_map, chunksize=50000):
        arr = chunk_df[gene_cols].values
        n_detected += (arr > 0).sum(axis=0).astype(np.int64)
        n_cells_total += arr.shape[0]
        del chunk_df, arr
        gc.collect()
    return dict(zip(gene_cols, n_detected / n_cells_total))


def detection_rates_paper07b():
    """Reuse the already-saved per-sample .npy blocks from run_paper07b_selfcomputed.py's
    extract() phase - avoids re-downloading/re-parsing the GEO tar."""
    parts_dir = PARTS_DIR / "07b_raw"
    gene_cols = (parts_dir / "gene_cols.txt").read_text().splitlines() \
        if (parts_dir / "gene_cols.txt").exists() else None
    npy_files = sorted(parts_dir.glob("*.npy"))
    assert npy_files, f"no .npy parts found in {parts_dir} - run run_paper07b_selfcomputed.py extract first"

    n_detected = None
    n_cells_total = 0
    for f in npy_files:
        arr = np.load(f)
        if n_detected is None:
            n_detected = np.zeros(arr.shape[1], dtype=np.int64)
            if gene_cols is None:
                gene_cols = [f"gene_{i}" for i in range(arr.shape[1])]
        n_detected += (arr > 0).sum(axis=0)
        n_cells_total += arr.shape[0]
        del arr
        gc.collect()
    rates = n_detected / n_cells_total
    return dict(zip(gene_cols, rates))


# Which reader belongs to which dataset.
READERS = {
    "01_Trem2R47H_MERFISH": detection_rates_paper01,
    "03_CosMx_AmyloidPlaqueNiche": detection_rates_paper03,
    "04_SEAAD_MERFISH": detection_rates_paper04,
    "05_Xenium_PVInterneuron_RetrosplenialCortex": detection_rates_paper05,
    "06_Xenium_APOE4vAPOE3": detection_rates_paper06,
    "07b_MERFISH_PU1_LymphoidMicroglia": detection_rates_paper07b,
    "08_MERFISH_TcellMyelin": detection_rates_paper08,
}
