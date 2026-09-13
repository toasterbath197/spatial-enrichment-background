"""
Every per-dataset fact, defined once.

These facts used to live in two places that had no way of knowing about each
other: each driver script stamped platform / species / brain_region onto its own
DE table, and build_dataset_table.py held a second copy for Table 1. They drifted.
Dataset 04's platform label ended up with three different values - one in the
driver, one in the stored DE table (never regenerated after the label changed),
and a third in Table 1.

So the facts live here now, and both consumers read from this one place:

    stamp(df, "04_SEAAD_MERFISH")     -> adds the DE-table columns
    DATASETS["04_SEAAD_MERFISH"]       -> the full record, for Table 1

Panel size is deliberately NOT recorded here. It is measured from the data every
time (build_dataset_table.py counts genes in the DE table) rather than trusted
from documentation - that is exactly how the control-probe contamination in
datasets 04 and 05 was caught. A number that can be measured should never be
written down twice.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# The project's standard DE method, used by every dataset except the two with
# documented exceptions (04 is a continuous covariate; 08 replicates a published
# comparison including its Bonferroni correction). Written once because it was
# previously copy-pasted into five drivers.
STANDARD_METHOD = ("cell_level_wilcoxon_rank_sum__normalized_counts__bh "
                   "(project standard, see project notes sec 2)")


@dataclass(frozen=True)
class Dataset:
    """One published study, as this project uses it."""

    # --- stamped onto every row of the DE table ---
    platform: str
    species: str
    brain_region: str
    source_file: str
    method: str = STANDARD_METHOD
    source: str = "self_computed"

    # --- descriptive, used only for Table 1 (datasets and covariates) ---
    animal_model_or_genotype: str = ""
    disease_stage_or_age: str = ""
    platform_description: str = ""
    comparison_type: str = ""
    # Set only where Table 1 needs a more specific region than the DE table's.
    table1_brain_region: str = ""

    # --- the author-published secondary tier, where one exists ---
    author_tier: "Dataset | None" = field(default=None, repr=False)

    @property
    def species_common(self) -> str:
        """Just "Mouse" or "Human", with any model detail stripped.

        The DE tables carry the model in the species string ("Mouse (5xFAD)")
        because a gene list is only interpretable alongside the model it came
        from. Table 1 wants the bare species, since it has a separate
        animal_model_or_genotype column - repeating the model in both would be
        the same fact printed twice in one row.
        """
        return self.species.split("(")[0].strip()

    @property
    def region_for_table1(self) -> str:
        return self.table1_brain_region or self.brain_region


DATASETS: dict[str, Dataset] = {

    "01_Trem2R47H_MERFISH": Dataset(
        platform="MERFISH (Vizgen VZG171, 300-gene)",
        species="Mouse",
        brain_region="Cortex, hippocampus, whole hemisections",
        source_file="MERFISH_Data.h5ad (layers/RNA, raw counts)",
        animal_model_or_genotype="WT / Trem2-R47H / 5xFAD / Trem2-R47H;5xFAD (4-way genotype cross)",
        disease_stage_or_age="Not specified in manifest/README - see Johnston et al. 2024 Methods",
        platform_description="MERFISH (Vizgen VZG171 commercial catalog panel + custom additions)",
        comparison_type="Genotype: WT vs Trem2R47H vs 5xFAD vs Trem2R47H;5xFAD",
    ),

    "03_CosMx_AmyloidPlaqueNiche": Dataset(
        platform="CosMx (NanoString/Bruker, 950-plex Mouse Neuroscience Panel)",
        species="Mouse (AppNL-G-F KI)",
        brain_region="Hippocampus and adjacent regions",
        source_file="GSE263791_RAW.tar (exprMat_file.csv.gz + metadata_file.csv.gz, Mean.BetaAmyloid)",
        animal_model_or_genotype="AppNL-G-F knock-in (amyloidosis model, no transgene overexpression)",
        disease_stage_or_age="18 months",
        platform_description="CosMx (NanoString/Bruker Mouse Neuroscience Panel)",
        comparison_type="Plaque-proximal vs distal (Mean.BetaAmyloid quartiles)",
    ),

    # Exception 1: the "DE" here is a continuous donor-level score, not a group
    # comparison, so the method string differs by design (project notes sec 3).
    "04_SEAAD_MERFISH": Dataset(
        platform="MERFISH (Allen SEA-AD panel: 140 genes; 40 Blank negative-control probes excluded)",
        species="Human",
        brain_region="Middle temporal gyrus (MTG)",
        source_file="SEAAD_MTG_MERFISH.2024-12-11.h5ad (obs: Donor ID, Continuous Pseudo-progression Score)",
        method="pseudobulk_mean_per_donor__spearman_correlation_with_CPS__bh (rank-based, updated 2026-08-04)",
        animal_model_or_genotype="n/a - human postmortem tissue, not an animal model",
        disease_stage_or_age="Continuous Pseudoprogression Score (CPS) per donor - not a discrete stage by design",
        platform_description="MERFISH (Allen SEA-AD panel: 140 genes; 40 Blank negative-control probes excluded)",
        comparison_type="Pseudobulk expression vs donor-level CPS (Spearman correlation)",
    ),

    "05_Xenium_PVInterneuron_RetrosplenialCortex": Dataset(
        platform="Xenium (10x Genomics Mouse Brain Panel: 247 genes; 294 control probes excluded)",
        species="Mouse (5xFAD)",
        brain_region="Retrosplenial cortex",
        source_file="GSE277463_RAW.tar (matrix.mtx.gz per sample)",
        animal_model_or_genotype="5xFAD",
        disease_stage_or_age="Not specified in manifest/README - see Terstege/Epp et al. 2025 Methods",
        platform_description="Xenium (10x Genomics Mouse Brain Panel: 247 genes; 294 control probes excluded)",
        comparison_type="Genotype (5xFAD vs WT) x Sex",
    ),

    "06_Xenium_APOE4vAPOE3": Dataset(
        platform="Xenium (10x Genomics, 266-gene panel)",
        species="Human",
        brain_region="Cortex",
        source_file="merged_xenium.h5ad (layers/counts, raw counts)",
        animal_model_or_genotype="n/a - human postmortem tissue; APOE3/APOE3 vs APOE4/APOE4 homozygous donors (3 each)",
        disease_stage_or_age="Not specified in manifest/README - see Millet/Ledo/Tavazoie et al. 2024 Methods",
        platform_description="Xenium (10x Genomics, 266-gene panel)",
        comparison_type="APOE3 vs APOE4",
    ),

    # The only dataset with both tiers. The author table describes a different
    # comparison from our own (plaque-proximal vs distal, not FADPU vs FADTV),
    # which is why its brain_region and platform labels differ - see project
    # notes sec 2 on why the two tiers are never substituted for one another.
    "07b_MERFISH_PU1_LymphoidMicroglia": Dataset(
        platform="MERFISH (398-gene panel)",
        species="Mouse (5xFAD; PU.1-low/high genetic manipulation)",
        brain_region="Cortex",
        source_file="GSE275026_RAW.tar (counts_and_metadata CSVs per sample, subsampled)",
        animal_model_or_genotype="5xFAD background; PU.1-low/high genetic manipulation (FADPU vs FADTV conditions)",
        disease_stage_or_age="Not specified in manifest/README - see Ayata/Schaefer et al. 2025 Methods",
        platform_description="MERFISH (targeted 398-gene panel)",
        comparison_type=("Self-computed tier: FADPU vs FADTV condition. "
                         "Author tier: plaque-associated vs distal microglia"),
        # Table 1 names the sub-population this dataset's cells were drawn from;
        # the DE table's brain_region stays the plain anatomical "Cortex".
        table1_brain_region="Cortex (plaque-proximal vs distal microglia)",
        author_tier=Dataset(
            platform="MERFISH (targeted panel)",
            species="Mouse (5xFAD; PU.1-low/high genetic manipulation)",
            brain_region="Cortex (plaque-proximal vs distal microglia)",
            source_file=("2024-08-16775C-s3/Supplementary_Table_4_-_MERFISH_from_"
                         "5xFAD-PU.1-low-wt-high_microglia.xlsx"),
            method="author_supplementary_table (Seurat FindMarkers, Wilcoxon rank-sum)",
            source="author_published",
        ),
    ),

    # Exception 2: reproduces a specific published figure, including that
    # figure's Bonferroni correction rather than the project's BH default.
    "08_MERFISH_TcellMyelin": Dataset(
        platform="MERFISH (497-gene panel)",
        species="Mouse (5xFAD)",
        brain_region="Half-brain sections",
        source_file="GSE243120_MERFISH_5xFAD_expression.txt.gz + _metadata.txt.gz",
        method=("cell_level_wilcoxon_rank_sum__normalized_counts__bonferroni "
                "(exact Fig 2h replication: 50-NN per Tcell, Bonferroni per paper's own legend - "
                "NOT the project-wide BH standard, deliberately)"),
        animal_model_or_genotype="5xFAD",
        disease_stage_or_age=("Not specified in manifest/README - see paper's Methods "
                              "(Nat Neurosci 2024, DOI 10.1038/s41593-024-01682-8)"),
        platform_description="MERFISH (497-gene panel per manifest; 496 genes measured in DE output, see NOTE)",
        comparison_type=("T-cell 50-nearest-neighbor vs non-neighbor "
                         "(exact Fig 2h replication, within 5xFAD only)"),
    ),
}


def stamp(df, paper_id: str, tier: str = "self_computed"):
    """Add the provenance columns every DE table carries, in place.

    Replaces the seven-line block each driver used to repeat:

        de["paper_id"] = ...; de["platform"] = ...; de["species"] = ... etc.

    tier: "self_computed" (default) or "author_published", which for dataset 07b
    selects the author table's own labels instead.
    """
    record = DATASETS[paper_id]
    if tier == "author_published":
        if record.author_tier is None:
            raise ValueError(f"{paper_id} has no author-published tier")
        record = record.author_tier

    df["paper_id"] = paper_id
    df["platform"] = record.platform
    df["species"] = record.species
    df["brain_region"] = record.brain_region
    df["source"] = record.source
    df["method"] = record.method
    df["source_file"] = record.source_file
    return df
