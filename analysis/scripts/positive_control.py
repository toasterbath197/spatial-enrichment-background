"""
Positive control — can the panel-specific background detect enrichment that is
really there?

The baseline experiment shows the panel background returns almost no significant
pathways. That result is ambiguous on its own: a background that returns nothing
whatever the input would produce the same table. This experiment removes the
ambiguity by planting a known enrichment and checking that it is recovered.

Design, per dataset:

  1. Take the dataset's real measured panel as the background.
  2. Pick target pathways that are testable within that panel (>= 8 panel genes).
  3. Build a query list of fixed size (40 genes, matching the synthetic
     simulation in Section 3.2) containing exactly k genes of the target
     pathway and 40 - k genes drawn at random from the rest of the panel.
  4. Run the identical enrichment engine used everywhere else, against both the
     panel background and the genome-wide background.
  5. Record whether the planted pathway is recovered, and how many OTHER
     pathways come out significant — those are false positives by construction,
     since the only enrichment placed in the query is the target's.

k = 0 is an internal null: the query is then a random draw of panel genes with
no planted signal.

Targets are required to have at least 15 panel genes — the largest spike level —
so that the same set of target pathways is used at every k. The dose-response is
therefore within-target, not confounded by targets dropping out as k rises.

Dataset 04 (SEA-AD) is excluded, as it is everywhere else in the paper: only one
pathway on its 140-gene panel meets the target size requirement.

This yields, for each background, both halves of the picture the baseline cannot
give on its own: recall (is real signal found?) and precision (how much else is
falsely reported alongside it?).

Output: fix11_positive_control_reactome.csv          one row per draw
        fix11_positive_control_summary_reactome.csv  aggregated over draws
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from de_pipeline import enrichment as en
from de_pipeline.paths import ENRICHMENT_DIR, MASTER_DE_TABLE

PADJ = 0.05
QUERY_SIZE = 40           # same query size as the synthetic simulation
MIN_TARGET_PANEL_GENES = 15   # >= max spike level, so every target supports every k
SPIKE_LEVELS = [0, 2, 4, 6, 8, 10, 12, 15]
N_TARGETS = 8
N_DRAWS = 15
SEED = 0


def panels_by_dataset():
    """One measured panel per dataset, from the same DE table every other
    experiment reads."""
    de = pd.read_csv(MASTER_DE_TABLE)
    de = de[de["source"] == "self_computed"]
    for paper_id, rows in de.groupby("paper_id"):
        yield paper_id, rows["species"].iloc[0], rows["gene"].unique()


def choose_targets(gmt, panel_annotated, rng, n_targets):
    """Pathways large enough inside this panel to plant a detectable signal in.

    Chosen at random among all qualifying pathways rather than by size, so the
    result does not depend on picking favourable targets.
    """
    qualifying = []
    for term_id, (description, genes) in gmt.items():
        in_panel = genes & panel_annotated
        if MIN_TARGET_PANEL_GENES <= len(in_panel) <= en.MAX_TERM_SIZE:
            qualifying.append((term_id, description, frozenset(in_panel)))
    qualifying.sort(key=lambda t: t[0])           # deterministic order first
    if len(qualifying) <= n_targets:
        return qualifying
    idx = rng.choice(len(qualifying), size=n_targets, replace=False)
    return [qualifying[i] for i in sorted(idx)]


def score(results, target_id):
    """Was the planted pathway recovered, and what else came out significant?"""
    if len(results) == 0:
        return dict(target_tested=False, target_sig=False, target_padj=np.nan,
                    n_sig=0, n_other_sig=0)
    hit = results[results["term_id"] == target_id]
    tested = len(hit) > 0
    padj = float(hit["padj"].iloc[0]) if tested else np.nan
    sig = bool(tested and padj < PADJ)
    n_sig = int((results["padj"] < PADJ).sum())
    return dict(target_tested=tested, target_sig=sig, target_padj=padj,
                n_sig=n_sig, n_other_sig=n_sig - (1 if sig else 0))


def run(db="reactome", n_targets=N_TARGETS, n_draws=N_DRAWS, only=None):
    rng = np.random.default_rng(SEED)
    rows = []

    for paper_id, species_field, panel in panels_by_dataset():
        if only and paper_id != only:
            continue
        if paper_id.startswith("04_"):        # excluded throughout the paper
            continue
        species = en.species_key(species_field)
        gmt = en.load_gmt(species, db)
        universe = en.gmt_universe(species, db)
        panel_annotated = frozenset(panel) & universe

        targets = choose_targets(gmt, panel_annotated, rng, n_targets)
        print(f"{paper_id}: panel {len(panel)} ({len(panel_annotated)} annotated), "
              f"{len(targets)} target pathways", flush=True)

        for term_id, description, target_genes in targets:
            others = np.array(sorted(panel_annotated - target_genes))
            for k in SPIKE_LEVELS:
                if k > len(target_genes) or QUERY_SIZE - k > len(others):
                    continue
                pool = np.array(sorted(target_genes))
                for draw in range(n_draws):
                    spike = rng.choice(pool, size=k, replace=False) if k else np.array([], dtype=object)
                    fill = rng.choice(others, size=QUERY_SIZE - k, replace=False)
                    query = np.concatenate([spike, fill])

                    panel_res = en.hypergeometric_enrichment(query, panel, gmt)
                    genome_res = en.hypergeometric_enrichment(query, universe, gmt)

                    rows.append(dict(
                        paper_id=paper_id, species_key=species,
                        panel_size=len(panel), term_id=term_id,
                        description=description,
                        target_genes_in_panel=len(target_genes),
                        spike_k=k, draw=draw, query_size=QUERY_SIZE,
                        **{f"panel_{key}": value for key, value in score(panel_res, term_id).items()},
                        **{f"genome_{key}": value for key, value in score(genome_res, term_id).items()},
                    ))
            print(f"   {term_id} done ({len(rows)} rows)", flush=True)

    draws = pd.DataFrame(rows)
    ENRICHMENT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{only}" if only else ""
    draws_path = ENRICHMENT_DIR / f"fix11_positive_control{tag}_{db}.csv"
    draws.to_csv(draws_path, index=False)
    print(f"\nwrote {len(draws):,} draws -> {draws_path.name}")

    summary = (draws.groupby(["paper_id", "panel_size", "spike_k"])
               .agg(n_draws=("draw", "size"),
                    panel_recall=("panel_target_sig", "mean"),
                    genome_recall=("genome_target_sig", "mean"),
                    panel_other_sig=("panel_n_other_sig", "mean"),
                    genome_other_sig=("genome_n_other_sig", "mean"),
                    panel_total_sig=("panel_n_sig", "mean"),
                    genome_total_sig=("genome_n_sig", "mean"))
               .reset_index())
    summary_path = ENRICHMENT_DIR / f"fix11_positive_control_summary{tag}_{db}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote {len(summary):,} summary rows -> {summary_path.name}")
    return draws, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default="reactome")
    parser.add_argument("--targets", type=int, default=N_TARGETS)
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument("--only", default=None, help="run a single paper_id")
    args = parser.parse_args()
    run(args.db, args.targets, args.draws, args.only)
