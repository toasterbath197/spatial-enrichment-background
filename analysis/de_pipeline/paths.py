"""
Every path in the project, defined once.

Before this module existed, each of the nine driver scripts worked out the
project root for itself and appended "databases" to it. That meant nine copies
of the same fact, and a comment in each one claiming "if the datasets move,
only this line changes" - which was false, because there were nine of them.
When the datasets did move, all nine had to be edited.

Now they are here, once. Scripts bootstrap with two lines:

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

and then import whichever paths they need by name.

Nothing here is absolute: every path is derived from this file's own location,
so the project can be moved or cloned anywhere and still run.
"""
import pathlib

DE_PIPELINE_DIR = pathlib.Path(__file__).resolve().parent
ANALYSIS_DIR = DE_PIPELINE_DIR.parent
PROJECT_ROOT = ANALYSIS_DIR.parent

# The seven raw datasets sit as sibling folders at the project root, one per
# published study (01_Trem2R47H_MERFISH, 03_CosMx_AmyloidPlaqueNiche, ...).
# Large, not in version control.
#
# THIS IS THE ONE LINE TO CHANGE if they are ever moved - for example, if they
# are grouped under a subfolder, this becomes PROJECT_ROOT / "<subfolder>".
DATA_ROOT = PROJECT_ROOT

SCRIPTS_DIR = ANALYSIS_DIR / "scripts"
PATHWAY_DB_DIR = ANALYSIS_DIR / "pathway_db"

RESULTS_DIR = ANALYSIS_DIR / "results"
DE_TABLES_DIR = RESULTS_DIR / "DE_tables"
ENRICHMENT_DIR = RESULTS_DIR / "enrichment_tables"

# Per-cell-type and per-sample intermediates for the datasets too large to
# process in one pass. Safe to delete; kept so a re-run can resume.
PARTS_DIR = DE_TABLES_DIR / "_parts"

MASTER_DE_TABLE = DE_TABLES_DIR / "master_DE_table.csv"
COVARIATE_TABLE = RESULTS_DIR / "covariate_table.csv"

FIGURES_DIR = PROJECT_ROOT / "paper_figures" / "figures"
TABLES_DIR = PROJECT_ROOT / "paper_figures" / "tables"

NOTES_FILE = PROJECT_ROOT / "00_PROJECT_NOTES_AND_METHODOLOGY.txt"


def dataset(folder_name: str, *parts) -> pathlib.Path:
    """A file inside one dataset's folder.

        dataset("08_MERFISH_TcellMyelin", "GSE243120_expression.txt.gz")

    Equivalent to DATA_ROOT / folder_name / *parts, but named so the intent is
    obvious at the call site and so every raw-data access is greppable.
    """
    return DATA_ROOT.joinpath(folder_name, *parts)
