"""Project paths, resolved from this file's location.

Every module and notebook resolves paths through here, so nothing depends on the
working directory. A notebook run from notebooks/ and a test run from the project
root both see the same PROJECT_ROOT.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA = PROJECT_ROOT / 'data'
RAW = DATA / 'raw'
INTERIM = DATA / 'interim'
PROCESSED = DATA / 'processed'

OUTPUTS = PROJECT_ROOT / 'outputs'
FIGURES = OUTPUTS / 'figures'
TABLES = OUTPUTS / 'tables'
DASHBOARDS = OUTPUTS / 'dashboards'

REPORTS = PROJECT_ROOT / 'reports'
NOTEBOOKS = PROJECT_ROOT / 'notebooks'


def ensure_dirs():
    """Create the output directories if a fresh clone is missing them."""
    for d in (RAW, INTERIM, PROCESSED, FIGURES, TABLES, DASHBOARDS, REPORTS):
        d.mkdir(parents=True, exist_ok=True)
