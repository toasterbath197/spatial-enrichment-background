"""
Paper 7b - MERFISH PU.1 / Lymphoid Microglia (author-published - CONFIRMED,
see manifest: 41586_2025_9662_MOESM3_ESM.zip contains
"Supplementary_Table_4_-_MERFISH_from_5xFAD-PU.1-low-wt-high_microglia.xlsx",
which has a sheet literally named "DEGs between plaque-associated " covering
exactly the MERFISH comparison this project needs (plaque-associated <15um
vs. distal >15um microglia, Wilcoxon rank-sum, matches manifest's
Comparison Type field). This is the primary DE source for paper 7b - no
self-computation needed.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import zipfile

from de_pipeline import author_tables, datasets, schema
from de_pipeline.paths import DATA_ROOT, DE_TABLES_DIR


ZIP = str(DATA_ROOT / "07b_MERFISH_PU1_LymphoidMicroglia" / "41586_2025_9662_MOESM3_ESM.zip")
MEMBER = "2024-08-16775C-s3/Supplementary_Table_4_-_MERFISH_from_5xFAD-PU.1-low-wt-high_microglia.xlsx"
PAPER_ID = "07b_MERFISH_PU1_LymphoidMicroglia"


def main():
    with zipfile.ZipFile(ZIP) as zf:
        raw = zf.read(MEMBER)
    tmp_path = "/tmp/paper07b_table4.xlsx"
    with open(tmp_path, "wb") as f:
        f.write(raw)

    de = author_tables.parse_author_de_table(
        tmp_path,
        sheet_name="DEGs between plaque-associated ",
        header_row=2,  # row 1 is a title, row 2 is the real header (Gene, avg_log2FC, ...)
        gene_col="Gene",
        log2fc_col="avg_log2FC",
        pval_col="p_val",
        padj_col="p_val_adj",
    )
    de["cell_type"] = "Microglia"
    de["comparison_type"] = "plaque-associated (<15um) vs distal (>15um) microglia"
    de["list_id"] = f"{PAPER_ID}__plaque_assoc_vs_distal__Microglia"
    datasets.stamp(de, PAPER_ID, tier="author_published")
    de["n_group1"] = None
    de["n_group2"] = None

    out = schema.make_df(de.to_dict("records"))
    outpath = str(DE_TABLES_DIR / "07b_MERFISH_PU1_DE.csv")
    schema.write_table(out, outpath)
    return out


if __name__ == "__main__":
    main()
