"""Parser for author-published supplementary DE tables (xlsx)."""

from __future__ import annotations
import pandas as pd
import openpyxl


def parse_author_de_table(xlsx_path: str, sheet_name: str, header_row: int,
                           gene_col: str, log2fc_col: str, pval_col: str,
                           padj_col: str | None = None) -> pd.DataFrame:
    """Read one sheet of an author-published supplementary table into a
    standardized (gene, log2FC, pvalue, padj) frame.

    header_row is 1-indexed (as in the spreadsheet), matching where the real
    column headers (Gene, avg_log2FC, ...) live - many supplementary tables
    have a title row above the real header.
    """
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb[sheet_name]
    rows = list(ws.iter_rows(min_row=header_row, values_only=True))
    header = rows[0]
    data = rows[1:]
    df = pd.DataFrame(data, columns=header)
    out = pd.DataFrame({
        "gene": df[gene_col],
        "log2FC": pd.to_numeric(df[log2fc_col], errors="coerce"),
        "pvalue": pd.to_numeric(df[pval_col], errors="coerce"),
        "padj": pd.to_numeric(df[padj_col], errors="coerce") if padj_col else None,
    })
    out = out.dropna(subset=["gene"])
    return out
