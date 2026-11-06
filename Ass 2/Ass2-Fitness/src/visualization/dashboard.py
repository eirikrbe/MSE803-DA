"""The fitness-app dashboard: one self-contained HTML page in the Ako Kupu design.

Four views, in the order a reader needs them:

  Overview     the validation outcome, the headline numbers and the findings (01, 02)
  Engagement   the structure of engagement and the twelve demographic tests (02, 03)
  Models       the three models against their baselines and structural benchmarks, and the two traps (04)
  Ethics       re-identification, differential privacy, subgroup audit, label validity, guidelines (05)

Aggregates only, by design (guideline G6 in notebook 05): the page carries the results tables and one
binned count of users, with any cell under MIN_CELL users suppressed. No row-level record reaches it.
It draws its charts as inline SVG and loads no script from outside the file; only the two web fonts
are fetched, and system fonts stand in offline.

    build_dashboard_html()                               notebook 05 calls this
    python -m src.visualization.dashboard --screenshot   also writes PNGs of the views (needs Chrome)
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.fitness import TIERS
from src.paths import DASHBOARDS, INTERIM, PROCESSED, TABLES

TEMPLATE = Path(__file__).with_name('dashboard_template.html')
NAME = 'fitness_dashboard'
VIEWS = ['overview', 'engagement', 'models', 'ethics']
MIN_CELL = 5                     # suppress any published count smaller than this


def suppress_small(counts, min_cell=MIN_CELL):
    """Replace counts below `min_cell` (but above zero) with None, so a small group is never published.
    Zeros stay zero: an empty cell reveals nothing about anyone."""
    out = []
    for row in counts:
        out.append([None if 0 < int(c) < min_cell else int(c) for c in row])
    return out


def _t(name):
    return pd.read_csv(TABLES / f'{name}.csv')


def _rows(df):
    return json.loads(df.to_json(orient='records'))          # NaN -> null, numpy -> plain


def _density():
    """Users per (sessions, calories) cell: 15 bands of 10 sessions x 18 bands of 50 kcal, small cells suppressed."""
    users = pd.read_csv(PROCESSED / 'users.csv')
    s_edges = np.arange(50, 201, 10)
    c_edges = np.arange(100, 1001, 50)
    h, _, _ = np.histogram2d(users['app_sessions'], users['calories'], bins=[s_edges, c_edges])
    return {'s_edges': s_edges.tolist(), 'c_edges': c_edges.tolist(), 'counts': suppress_small(h.T.tolist()),
            'min_cell': MIN_CELL}


def _confusion():
    p = pd.read_csv(INTERIM / '04_test_predictions.csv')
    cm = pd.crosstab(p['activity_level'], p['tier_pred']).reindex(index=TIERS, columns=TIERS, fill_value=0)
    return {'labels': TIERS, 'counts': cm.to_numpy().tolist()}


def dashboard_data():
    """Everything the page shows, computed from outputs/tables (and two aggregate counts)."""
    reg = _t('04_regression_test').merge(_t('04_regression_cv'), on='model')
    clf = _t('04_classification_test').merge(_t('04_classification_cv'), on='model', suffixes=('', '_cv'))
    kan = _t('05_k_anonymity')
    return {
        'meta': {'byline': 'Eric Gomez · November 2026', 'n_raw': 5299, 'n_users': 5000, 'n_distance': 4997,
                 'n_test': 1000, 'min_cell': MIN_CELL},
        'validation': _rows(_t('02_validation_matrix')),
        'flow': [{'label': 'Genuine users', 'n': 5000, 'key': 'kept'},
                 {'label': 'Verbatim copies', 'n': 272, 'key': 'copy'},
                 {'label': 'Altered copies', 'n': 27, 'key': 'alt'}],
        'tiers': _rows(_t('02_tier_summary')),
        'profile': _rows(_t('02_demographic_profile')),
        'sds': _rows(_t('02_descriptives')[['variable', 'std']]),
        'tests': _rows(_t('03_hypothesis_tests')),
        'density': _density(),
        'regression': _rows(reg),
        'intervals': _rows(_t('04_prediction_intervals')),
        'classification': _rows(clf),
        'mcnemar': _rows(_t('04_mcnemar')),
        'thresholds': _rows(_t('04_sedentary_thresholds')),
        'confusion': _confusion(),
        'kmeans': _rows(_t('04_kmeans_sweep')),
        'profiles': _rows(_t('04_kmeans_profiles')),
        'rule_segments': _rows(_t('04_rule_segments')),
        'inflation': _rows(_t('04_copy_inflation')),
        'sensitivity': _rows(_t('04_sensitivity')),
        'kanon': _rows(kan),
        'dp': _rows(_t('05_dp_error')),
        'fairness': _rows(_t('05_fairness_by_group')),
        'replication': _rows(_t('05_fairness_replication')),
        'label': _rows(_t('05_label_validity')),
        'guidelines': _rows(_t('05_guidelines')),
        'representation': _rows(_t('05_representation')),
    }


def render_html():
    """The page as one string: the template with the data embedded."""
    payload = json.dumps(dashboard_data(), ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    payload = payload.replace('</', '<\\/')                   # no string in the data can close the <script>
    return TEMPLATE.read_text(encoding='utf-8').replace('__DATA__', payload)


def build_dashboard_html(name=NAME):
    """Write the dashboard to outputs/dashboards/<name>.html and return its path."""
    DASHBOARDS.mkdir(parents=True, exist_ok=True)
    path = DASHBOARDS / f'{name}.html'
    path.write_text(render_html(), encoding='utf-8')
    print(f'saved outputs/dashboards/{path.name} ({path.stat().st_size / 1024:.0f} KB)')
    return path


def screenshot_dashboard(path=None, views=('overview',), size=(1320, 1040), scale=2):
    """PNG of each view in `views` with headless Chrome, for the README and the report. Needs a local Chrome."""
    path = Path(path or DASHBOARDS / f'{NAME}.html')
    chrome = next((c for c in ('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
                               shutil.which('google-chrome') or '', shutil.which('chromium') or '')
                   if c and Path(c).exists()), None)
    if chrome is None:
        raise FileNotFoundError('Chrome was not found; open the HTML file and take the screenshot by hand')
    out = []
    for view in views:
        png = path.with_name(f'{path.stem}_{view}.png')
        subprocess.run([chrome, '--headless=new', '--hide-scrollbars', '--disable-gpu',
                        f'--window-size={size[0]},{size[1]}', f'--force-device-scale-factor={scale}',
                        '--virtual-time-budget=6000', f'--screenshot={png}', f'{path.as_uri()}#{view}'],
                       check=True, capture_output=True)
        print(f'saved outputs/dashboards/{png.name}')
        out.append(png)
    return out


if __name__ == '__main__':
    import sys
    built = build_dashboard_html()
    if '--screenshot' in sys.argv:
        screenshot_dashboard(built, views=VIEWS)
