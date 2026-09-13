"""
Orchestrator: runs every paper whose driver is unblocked, concatenates their
standardized DE tables into one master long-format table, and prints a status
summary for all 8 manifest papers (ready / blocked / excluded).

NOTE ON RUNTIME: in this project's own analysis sandbox (very limited RAM,
45s-per-command budget), papers 1, 6, and 7b's self-computed tier were each
run as several separate smaller commands (one cell type / one sample at a
time - see each script's own --all vs per-unit usage in its docstring) rather
than through this orchestrator, because a single process doing all of it at
once was OOM-killed. On a normal machine (a few GB of free RAM, no per-command
time cap) this script should just work end to end - included for
completeness/reproducibility, not because it was how the results in
results/DE_tables/ were actually produced.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import subprocess
import pandas as pd
from de_pipeline import schema

from de_pipeline.paths import SCRIPTS_DIR, DE_TABLES_DIR, MASTER_DE_TABLE

# (script, args) - self-computed primary tier for every data-complete paper,
# plus paper 7b's author-published secondary tier.
READY = [
    ("run_paper01_trem2.py", ["--all"]),
    ("run_paper03_cosmx.py", []),
    ("run_paper04_seaad.py", []),
    ("run_paper05_xenium.py", []),
    ("run_paper06_apoe.py", ["--all"]),
    ("run_paper07b_author_table.py", []),           # secondary/robustness tier
    ("run_paper07b_selfcomputed_all.py", []),        # primary tier (wraps extract x8 + finalize)
    ("run_paper08_tcellmyelin.py", []),
]
BLOCKED: list[str] = []  # nothing left blocked as of 2026-08-04 (paper 6 unblocked)
EXCLUDED = ["02_MicrogliaXenium_preprint - deprioritized, no public data deposit (see README)"]

STATUS = {
    "01_Trem2R47H_MERFISH": "READY - self-computed (cell-level Wilcoxon, genotype x cell type)",
    "02_MicrogliaXenium_preprint": "EXCLUDED - no public data deposit, deprioritized",
    "03_CosMx_AmyloidPlaqueNiche": "READY - self-computed (cell-level Wilcoxon, plaque-proximity bins)",
    "04_SEAAD_MERFISH": "READY - self-computed (Spearman correlation with CPS, by design)",
    "05_Xenium_PVInterneuron_RetrosplenialCortex": "READY - self-computed (cell-level Wilcoxon, genotype)",
    "06_Xenium_APOE4vAPOE3": "READY - self-computed (cell-level Wilcoxon, APOE4 vs APOE3 x cell type) - "
                              "unblocked 2026-08-04 via merged_xenium.h5ad from Zenodo 10.5281/zenodo.8206638",
    "07b_MERFISH_PU1_LymphoidMicroglia": "READY - BOTH tiers: self-computed primary (FADPU vs FADTV) "
                                          "+ author-published secondary (Supplementary Table 4, plaque-based)",
    "08_MERFISH_TcellMyelin": "READY - self-computed, exact Fig 2h replication (50-NN per Tcell, Bonferroni)",
}


def main():
    DE_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for script, args in READY:
        print(f"\n=== running {script} {' '.join(args)} ===")
        subprocess.run([sys.executable, "-u", str(SCRIPTS_DIR / script), *args], check=True)

    csvs = sorted(DE_TABLES_DIR.glob("*_DE.csv"))
    dfs = [pd.read_csv(c) for c in csvs]
    master = pd.concat(dfs, ignore_index=True)
    schema.write_table(master, MASTER_DE_TABLE)

    print("\n=== PIPELINE STATUS (all 8 manifest papers) ===")
    for paper, status in STATUS.items():
        print(f"  {paper:50s} {status}")

    print(f"\nMaster table: {len(master):,} rows across {master['paper_id'].nunique()} papers, "
          f"{master['list_id'].nunique()} distinct gene lists.")
    return master


if __name__ == "__main__":
    main()
