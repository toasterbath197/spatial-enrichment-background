# Background gene set choice in targeted spatial transcriptomics

Analysis code for the manuscript *"Background gene set choice inflates pathway enrichment
significance in targeted spatial transcriptomics of Alzheimer's disease."*

**The finding.** Pathway over-representation analysis needs a background gene set — the universe
your genes could have been drawn from. It enters the hypergeometric test as `N`, in the
denominator of the expected overlap `nK/N`. The enrichment tools in widest use do not ask for it:
g:Profiler defaults to `domain_scope="annotated"` and clusterProfiler's `enricher()` falls back to
every gene in the supplied annotation when `universe` is omitted. For whole-transcriptome RNA-seq
that default is approximately right. For a targeted spatial panel measuring a few hundred genes it
is wrong by an order of magnitude.

Reanalysing seven published Alzheimer's spatial transcriptomics datasets under one standardised
pipeline, the default background produced significant pathways on all 49 mouse gene lists (mean
54.67 per list) where the correct panel-specific background produced them on one (mean 0.041).
A negative control shows the excess to be false positives on 91.8% of lists against 2.0%; a
positive control shows the panel background nonetheless recovers planted enrichment in 100% of
draws at the highest spike level.

## Layout

```
analysis/de_pipeline/     importable library: DE statistics, enrichment engine, IO, schema
analysis/scripts/         one script per dataset, plus the experiment drivers
analysis/tests/           unit tests for the enrichment engine
analysis/verify_results.py  re-checks published numbers against the result tables
paper_figures/            figure builders (read result tables, compute nothing)
```

Manuscript build scripts are not included: this repository is the analysis, not the
typesetting. Every number in the paper comes from the code here.

The two entry points that matter:

```bash
# every enrichment experiment in the paper
python3 analysis/scripts/run_experiments.py all

# the positive control (run one dataset at a time; each takes 30-180s)
python3 analysis/scripts/positive_control.py --only 01_Trem2R47H_MERFISH
```

`run_experiments.py --help` lists the individual experiments: `baseline`, `simulation`,
`subsampling`, `negative-control`, `detected-background`, `regression`, `sensitivity`.

## Reproducing from scratch

1. **Install dependencies.** `pip install -r requirements.txt` (Python 3.10.12 was used).

2. **Obtain the data.** Not redistributed here — all seven datasets are publicly deposited.
   Accessions are listed in Methods 2.2 of the manuscript: Brain Image Library (01),
   GSE263791 (03), Allen Institute Brain Cell Atlas release 2024-12-11 (04), GSE277463 (05),
   Zenodo 10.5281/zenodo.8206638 (06), GSE275026 (07b), GSE243120 (08). Place each under the
   numbered folder its script expects (see `analysis/de_pipeline/paths.py`).

3. **Obtain the pathway definitions.** Reactome and WikiPathways gene-symbol GMT files come from
   the companion data repository of the `mulea` R package
   (`ELTEbioinformatics/GMT_files_for_mulea`, commit `601ed08c`), placed in
   `analysis/pathway_db/`. They are not redistributed here. Note that the GO Biological Process
   files from that repository have a construction defect documented in
   `analysis/de_pipeline/enrichment.py` and must not be used.

4. **Run the DE pipeline**, then the experiments, then the figures.

All random draws use `numpy.random.default_rng` with a fixed seed, so every reported number is
reproducible exactly.

## Cross-references in the code

Docstrings and comments throughout refer to `00_PROJECT_NOTES_AND_METHODOLOGY.txt`, a running
methodology log kept during the work, and occasionally to other internal documents. These are not
published: the log is a working record, including dead ends and superseded decisions, rather than a
description of the final analysis. The manuscript's Methods section is its public equivalent, and
every method referenced by a section number in the code is described there. Nothing needed to run
or understand the code depends on them.

## Result tables

The generated tables are excluded from this repository to keep it small. Every quantitative claim
in the manuscript is read from a named CSV under `analysis/results/enrichment_tables/`; the mapping
from claim to file is given in the manuscript's companion document. Those tables are archived with
the dataset DOI rather than here.

## A note on the pathway size filter

Pathways with fewer than 5 or more than 500 genes *within the background under test* are excluded
before correction. A pathway's effective size therefore changes with the background, by definition
— this is deliberate and is discussed in the manuscript. It has one consequence worth knowing
before reading the positive-control output: pathways exceeding 500 genes genome-wide are dropped
under the genome background while remaining testable within a panel, so the two arms are compared
on the subset both can test.

## Citation

Manuscript in preparation. Citation details to follow.

## License

MIT — see `LICENSE`.
