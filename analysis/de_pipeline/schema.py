"""
Standardized long-format DE table schema shared across all 8 papers in the
background-gene-set-bias project.

One row = one gene, in one comparison, in one paper (optionally, one cell type).
This is the "tidy long-format table" called for in Phase 3 of the pre-analysis
plan (list_id, background_type, term/gene, p-adjusted, etc.) - background_type
is deliberately NOT filled in here: that gets attached later at the enrichment
stage (Phase 3/4), once genome vs. panel background is chosen. This module only
produces the DE gene lists themselves.
"""

from __future__ import annotations
import pathlib

import pandas as pd

COLUMNS = [
    "paper_id",           # e.g. "01_Trem2R47H_MERFISH"
    "list_id",             # unique per gene list, e.g. "01_5xFAD_vs_WT__microglia"
    "platform",            # MERFISH / CosMx / Xenium
    "species",
    "brain_region",
    "comparison_type",     # human-readable, e.g. "5xFAD vs WT"
    "cell_type",           # "all" if not cell-type-resolved
    "gene",
    "log2FC",
    "pvalue",
    "padj",
    "n_group1",
    "n_group2",
    "source",              # "self_computed" or "author_published"
    "method",              # e.g. "pseudobulk_welch_ttest_bh", "author_supplementary_table"
    "source_file",         # provenance: which file this came from
]


def make_df(rows: list) -> pd.DataFrame:
    """Build a schema-conformant DataFrame from a list of row dicts, filling
    any missing columns with None."""
    df = pd.DataFrame(rows)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[COLUMNS]


def write_table(df: pd.DataFrame, path) -> None:
    """Write a DE table, refusing anything that isn't schema-conformant.

    Creates the destination directory if it doesn't exist. Every driver used to
    carry its own `os.makedirs(os.path.dirname(outpath))` line immediately before
    calling this; there were eight identical copies. The function that does the
    writing is the right place for it.
    """
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"DE table missing required columns: {missing}")
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_csv(path, index=False)
    print(f"  wrote {len(df):,} rows -> {path}")
