"""Tests for src/visualization/dashboard.py.

The first two tests use made-up sentences. The rest check the page against this project's own
processed data and results tables, so run notebooks 02 and 03 first:
    data_env/bin/python3 tests/test_dashboard.py

pytest is not installed in data_env, so these are plain functions with plain asserts.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import PROCESSED, TABLES  # noqa: E402
from src.visualization.dashboard import BANDS, dashboard_data, impact_numbers, render_html  # noqa: E402

DATA = dashboard_data()


# --------------------------------------------------------------------------
# impact_numbers - reads 'before → after' out of a matrix sentence
# --------------------------------------------------------------------------

def test_impact_numbers_reads_the_first_pair():
    n = impact_numbers('December mean temperature 18.1 → 23.0 °C', 'Remove: altered copy of row 70')
    assert n == {'metric': 'December mean temperature', 'before': '18.1', 'after': '23.0', 'unit': '°C',
                 'alternative': 'If kept', 'more': False}
    n = impact_numbers('Visits below 7 mg/L 12 → 11 (AV-3 on 29 Dec: 6.9 → 9.9 mg/L)', 'Remove: no source row')
    assert (n['metric'], n['before'], n['after'], n['more']) == ('Visits below 7 mg/L', '12', '11', True)


def test_impact_numbers_names_the_alternative_and_skips_what_it_cannot_read():
    n = impact_numbers('Dropped instead: īnanga records 25 → 24, per survey ...', 'Infer īnanga from its size')
    assert (n['metric'], n['alternative']) == ('Īnanga records', 'If dropped')
    assert impact_numbers('pH summarised over 68 of 69 visits', 'Leave blank; no imputation') is None
    assert impact_numbers('Counts 3 → 4', 'Leave blank; no imputation') is None


# --------------------------------------------------------------------------
# the page against the project's data
# --------------------------------------------------------------------------

def test_counts_match_the_processed_data():
    visits = pd.read_csv(PROCESSED / 'visits.csv')
    records = pd.read_csv(PROCESSED / 'fish_records.csv')
    assert len(DATA['visits']) == DATA['river']['visits'] == len(visits)
    assert DATA['fish']['records'] == len(records)
    assert sum(s['records'] for s in DATA['fish']['species']) == len(records)


def test_every_band_follows_the_thresholds():
    for v in DATA['visits']:
        expected = next(b['band'] for b in BANDS if b['floor'] is None or v['do'] >= b['floor'])
        assert v['band'] == expected, (v['site'], v['date'], v['do'], v['band'])


def test_fish_means_match_figure_d_table():
    table = pd.read_csv(TABLES / '02_fish_by_month.csv', header=[0, 1], index_col=0)
    names = {s['code']: s['name'] for s in DATA['fish']['species']}
    for row in DATA['fish']['summary']['all']:
        name = names[row['species']]
        assert row['n'] == table.loc[name, ('size', row['month'])], (name, row['month'])
        assert abs(row['mean'] - table.loc[name, ('mean', row['month'])]) < 0.051, (name, row['month'])


def test_site_filters_add_up_to_all_sites():
    total = {(r['species'], r['month']): r['n'] for r in DATA['fish']['summary']['all']}
    for (species, month), n in total.items():
        per_site = sum(r['n'] for site in DATA['meta']['sites'] for r in DATA['fish']['summary'][site]
                       if (r['species'], r['month']) == (species, month))
        assert per_site == n, (species, month)


def test_validation_flow_adds_up():
    v = DATA['validation']
    for f in v['flow']:
        assert f['before'] == f['after'] + sum(p['n'] for p in f['parts']), f['table']
    assert v['raw'] == v['removed'] + v['merged'] + DATA['river']['visits'] + DATA['fish']['records']


def test_every_matrix_problem_with_numbers_is_read():
    for e in DATA['validation']['entries']:
        if '→' in e['impact'] and e['decision'].split()[0].rstrip(':;') in ('Remove', 'Infer'):
            assert e['numbers'] is not None, e['problem']


def test_page_is_self_contained():
    html = render_html()
    assert html.startswith('<!doctype html>')
    assert '<meta name="viewport"' in html
    assert '__DATA__' not in html
    assert not re.search(r'<script[^>]+src=', html), 'a script is loaded from outside the file'
    hrefs = re.findall(r'<link[^>]+href="([^"]+)"', html)
    assert all(h.startswith('https://fonts.') for h in hrefs), hrefs
    payload = html.split('window.DATA = ', 1)[1].split(';\n</script>', 1)[0]
    assert json.loads(payload.replace('<\\/', '</')) == DATA


if __name__ == '__main__':
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith('test_') and callable(f)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f'  PASS  {name}')
        except AssertionError as exc:
            failures += 1
            print(f'  FAIL  {name}: {exc}')
    print(f'\n{len(tests) - failures}/{len(tests)} passed')
    sys.exit(1 if failures else 0)
