"""
Minimal h5ad reader built directly on h5py (no anndata/scanpy dependency -
these files are multi-GB and we only ever need a handful of obs columns plus
X or one layer, so a full anndata load is unnecessary weight).

Only supports what these 8 papers' h5ad files actually use: dense or CSR X,
categorical (categories/codes) or plain obs columns, plain var/_index gene names.
"""

from __future__ import annotations
import h5py
import numpy as np
import pandas as pd


def read_var_names(path: str) -> list[str]:
    with h5py.File(path, "r") as f:
        raw = f["var/_index"][:]
        return [x.decode() if isinstance(x, bytes) else x for x in raw]


def read_obs_column(path: str, col: str) -> pd.Series:
    with h5py.File(path, "r") as f:
        grp = f[f"obs/{col}"]
        if isinstance(grp, h5py.Group):
            cats = grp["categories"][:]
            cats = [c.decode() if isinstance(c, bytes) else c for c in cats]
            codes = grp["codes"][:]
            vals = np.array([cats[c] if c >= 0 else None for c in codes], dtype=object)
            return pd.Series(vals, name=col)
        else:
            return pd.Series(grp[:], name=col)


def read_X(path: str, layer: str | None = None, dtype=np.float32) -> np.ndarray:
    """Read X (or a named layer) fully into memory as a dense array.
    Handles both a plain dense dataset and a CSR-encoded group."""
    with h5py.File(path, "r") as f:
        node = f[f"layers/{layer}"] if layer else f["X"]
        if isinstance(node, h5py.Dataset):
            return node.astype(dtype)[:]
        # CSR sparse group: data / indices / indptr
        from scipy.sparse import csr_matrix
        n_obs = len(f["obs/_index"])
        n_var = len(f["var/_index"])
        mat = csr_matrix(
            (node["data"][:], node["indices"][:], node["indptr"][:]),
            shape=(n_obs, n_var),
        )
        return np.asarray(mat.todense(), dtype=dtype)
