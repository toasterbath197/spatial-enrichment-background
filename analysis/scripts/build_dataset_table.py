"""
Fix 6a (00_PROJECT_NOTES_AND_METHODOLOGY.txt Section 9): build the covariate
table for the across-study (7-paper) comparison - one row per paper, columns
for species, genotype/animal model, disease stage/age, brain region,
platform, and panel size. Pure documentation of facts already established
during data acquisition (phase2_manifest.xlsx, each paper's README.txt) plus
panel size measured directly from the actual DE output (master_DE_table.csv)
rather than trusted from a manifest description string, since the manifest's
platform text ("140-gene panel" for paper 4, e.g.) turned out to disagree
with the panel actually measured in the data (180 genes) - see NOTE column.

No statistical adjustment happens here (that's Fix 6b, which depends on
Step 0's full baseline results). This step only makes every known
cross-paper difference visible in one place, per Fix 6's stated purpose:
"Makes every known difference between papers visible and citable, instead
of hidden." Where a field genuinely isn't recorded anywhere in this
project's own manifest/READMEs (e.g. exact animal age for most papers),
that gap is stated explicitly rather than guessed.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd

from de_pipeline import datasets

from de_pipeline.paths import MASTER_DE_TABLE, COVARIATE_TABLE

DE_TABLE = str(MASTER_DE_TABLE)
OUT_PATH = COVARIATE_TABLE

# Per-dataset facts come from de_pipeline/datasets.py, which is also what the
# driver scripts stamp onto the DE tables. There used to be a second copy of
# these facts right here, and the two drifted - see that module's docstring.
# Only fields not derivable from the DE table itself are recorded there; panel
# size is measured below rather than looked up.
MANUAL_COVARIATES = {
    paper_id: {
        "species": d.species_common,
        "animal_model_or_genotype": d.animal_model_or_genotype,
        "disease_stage_or_age": d.disease_stage_or_age,
        "brain_region": d.region_for_table1,
        "platform_description": d.platform_description,
        "comparison_type": d.comparison_type,
    }
    for paper_id, d in datasets.DATASETS.items()
}


def main():
    df = pd.read_csv(DE_TABLE)
    df = df[df["source"] == "self_computed"]

    rows = []
    for paper_id, g in df.groupby("paper_id"):
        manual = MANUAL_COVARIATES.get(paper_id, {})
        panel_size_measured = int(g["gene"].nunique())
        n_lists = int(g["list_id"].nunique())
        n_cell_types = int(g["cell_type"].nunique())
        platform_desc = manual.get("platform_description", g["platform"].iloc[0])

        note = ""
        if paper_id == "04_SEAAD_MERFISH" and panel_size_measured != 140:
            note = f"Manifest/platform text says '140-gene panel'; {panel_size_measured} genes actually measured in DE output - using measured value as source of truth."
        if paper_id == "08_MERFISH_TcellMyelin" and panel_size_measured != 497:
            note = f"Manifest says 497-gene panel; {panel_size_measured} genes actually measured in DE output - using measured value as source of truth."
        # RESOLVED 2026-08-12. The "248-gene panel" figure carried in this project's
        # notes and the dataset README was simply wrong. 10x Genomics' own pre-designed
        # panel table lists the Xenium Mouse Brain Gene Expression Panel at 247 genes
        # targeted, which is exactly what the deposit's features.tsv contains. The
        # vendor also documents 27 negative control targets for this panel specifically
        # (all other v1 panels have 20), and the deposit has 27 - independent
        # confirmation it is the same panel. Measured and documented now agree.
        if paper_id == "05_Xenium_PVInterneuron_RetrosplenialCortex" and panel_size_measured != 247:
            note = (f"Expected 247 genes (10x pre-designed Xenium Mouse Brain Panel, vendor-stated); "
                    f"{panel_size_measured} measured in DE output - investigate before citing.")

        rows.append({
            "paper_id": paper_id,
            "species": manual.get("species", g["species"].iloc[0]),
            "animal_model_or_genotype": manual.get("animal_model_or_genotype", ""),
            "disease_stage_or_age": manual.get("disease_stage_or_age", ""),
            "brain_region": manual.get("brain_region", g["brain_region"].iloc[0]),
            "platform": platform_desc,
            "panel_size_measured": panel_size_measured,
            "comparison_type": manual.get("comparison_type", g["comparison_type"].iloc[0]),
            "n_lists_de_comparisons": n_lists,
            "n_cell_types": n_cell_types,
            "note": note,
        })

    out = pd.DataFrame(rows).sort_values("panel_size_measured").reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"Wrote {len(out)} rows -> {OUT_PATH}\n")
    print(out[["paper_id", "species", "panel_size_measured", "brain_region"]].to_string(index=False))
    print()
    print("Full covariate table (species / genotype / disease stage / region / platform / panel size):")
    for _, r in out.iterrows():
        print(f"\n{r['paper_id']}")
        print(f"  species:              {r['species']}")
        print(f"  genotype/model:       {r['animal_model_or_genotype']}")
        print(f"  disease stage/age:    {r['disease_stage_or_age']}")
        print(f"  brain region:         {r['brain_region']}")
        print(f"  platform:             {r['platform']}")
        print(f"  panel size (measured):{r['panel_size_measured']}")
        print(f"  comparison type:      {r['comparison_type']}")
        if r["note"]:
            print(f"  NOTE:                 {r['note']}")

    return out


if __name__ == "__main__":
    main()
