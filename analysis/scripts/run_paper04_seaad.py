"""
Paper 4 - SEA-AD MERFISH (self-computed BY DESIGN - manifest note: "DE list to
be computed by joining on Donor ID, not a static pre-made table").

Data on disk: 04_SEAAD_MERFISH/SEAAD_MTG_MERFISH.2024-12-11.h5ad, 1,887,729
cells x 140-gene panel, MTG. obs already contains 'Continuous Pseudo-progression
Score' (CPS, donor-level AD severity) per cell - no separate join against the
CPS xlsx is actually needed, it's pre-merged into this object.

For each of the 24 annotated Subclasses (cell types), pseudobulk by Donor ID
(27 donors), then correlate pseudobulk expression with donor-level CPS score.
This is the exact "gene-by-pseudoprogression correlation" design described in
the manifest and pre-analysis-plan notes.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from de_pipeline import datasets, h5ad_io, schema, stats_utils
from de_pipeline.paths import DATA_ROOT, DE_TABLES_DIR


H5AD = str(DATA_ROOT / "04_SEAAD_MERFISH/SEAAD_MTG_MERFISH.2024-12-11.h5ad")
PAPER_ID = "04_SEAAD_MERFISH"


def main():
    print(f"[{PAPER_ID}] loading obs columns + X ...")
    genes = h5ad_io.read_var_names(H5AD)
    donor = h5ad_io.read_obs_column(H5AD, "Donor ID").values
    subclass = h5ad_io.read_obs_column(H5AD, "Subclass").values
    cps = pd.Series(h5ad_io.read_obs_column(H5AD, "Continuous Pseudo-progression Score").values.astype(float))
    X = h5ad_io.read_X(H5AD)  # raw MERFISH spot counts per cell (uns/X_normalization is a suggested transform, not pre-applied - values observed 0-58, consistent with raw counts)

    # The var index holds 180 features, of which 40 are "Blank-NN" MERFISH negative
    # control probes (barcodes deliberately assigned to no transcript, used to
    # estimate the false-detection rate). They are not genes and must never enter a
    # DE test or a background gene set. Dropping them leaves 140 - which is exactly
    # the panel size the manifest and the Allen SEA-AD documentation state, so the
    # long-standing "140 documented vs 180 measured" discrepancy (notes Section 22) was this contamination, not a documentation error.
    # Same class of bug as run_paper05_xenium.py's control-probe filter; see notes
    # Section 22.
    gene_mask = np.array([not str(g).lower().startswith("blank") for g in genes])
    n_dropped = int((~gene_mask).sum())
    genes = [g for g, keep in zip(genes, gene_mask) if keep]
    X = X[:, gene_mask]
    print(f"  dropped {n_dropped} Blank-* negative control probes -> {len(genes)} real genes")
    print(f"  X shape {X.shape}, {len(genes)} genes, {len(set(donor))} donors, {len(set(subclass))} subclasses")

    donor_cps = pd.Series(cps.values, index=donor).groupby(level=0).mean()

    all_rows = []
    for ct in sorted(set(subclass)):
        mask = subclass == ct
        if mask.sum() < 200:
            continue
        pb = stats_utils.build_pseudobulk(X[mask], genes, donor[mask], agg="mean")
        de = stats_utils.continuous_covariate_de(pb, donor_cps)
        de["cell_type"] = ct
        de["comparison_type"] = "Continuous Pseudoprogression Score (CPS) correlation"
        de["list_id"] = f"{PAPER_ID}__CPS_correlation__{ct.replace(' ', '_').replace('/', '-')}"
        all_rows.append(de)
        print(f"  done subclass: {ct} (n_donors_with_cells={pb.shape[0]})")

    de_all = pd.concat(all_rows, ignore_index=True)
    datasets.stamp(de_all, PAPER_ID)

    out = schema.make_df(de_all.to_dict("records"))
    outpath = str(DE_TABLES_DIR / "04_SEAAD_MERFISH_DE.csv")
    schema.write_table(out, outpath)
    return out


if __name__ == "__main__":
    main()
