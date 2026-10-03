"""The Avon River dashboard: one self-contained HTML page in the Ako Kupu design.

The page has four views, in the order a reader needs them:

  River check      every visit's water quality by site, against the indicative NPS-FM bands
  Fish             counts per survey by species, month and site
  Data validation  the four classifications and the computed impact of keeping each record (02)
  Findings         the six tests, the sensitivity analysis and the model check (03)

Every number on the page is computed here from data/processed/ and outputs/tables/, so the page
changes when the analysis does. The page draws its own charts as inline SVG and loads no script
from outside the file. Only the two web fonts are fetched, and system fonts stand in offline.

    build_dashboard_html()                               notebook 03 calls this
    python -m src.visualization.dashboard --screenshot   also writes a PNG of the River check view (needs Chrome)
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from src.paths import DASHBOARDS, PROCESSED, TABLES

TEMPLATE = Path(__file__).with_name('dashboard_template.html')
NAME = 'avon_river_dashboard'
VIEWS = ['river', 'fish', 'validation', 'findings']

SITES = ['AV-1', 'AV-2', 'AV-3']
SUMMER_STARTS = '2023-11-01'                 # NPS-FM summer window: 1 November to 30 April
SPECIES = {                                  # name in the data -> on the page; Figure D's order
    'Inanga': 'Īnanga', 'Longfin Eel': 'Longfin eel', 'Shortfin Eel': 'Shortfin eel', 'Brown Trout': 'Brown trout'}
BANDS = [                                    # NPS-FM 2020 Table 17 floors (mg/L) as in 02; Figure C's colours
    {'band': 'A', 'floor': 8.0, 'range': '≥ 8.0', 'meaning': 'no stress', 'fill': '#d1e5f0', 'ink': '#2b2b2b'},
    {'band': 'B', 'floor': 7.0, 'range': '7.0–8.0', 'meaning': 'minor stress', 'fill': '#92c5de', 'ink': '#2b2b2b'},
    {'band': 'C', 'floor': 5.0, 'range': '5.0–7.0', 'meaning': 'moderate stress', 'fill': '#e07b5f', 'ink': '#2b2b2b'},
    {'band': 'D', 'floor': None, 'range': '< 5.0', 'meaning': 'significant stress', 'fill': '#b2182b', 'ink': '#ffffff'},
]
PH_GUIDELINE = [7.2, 7.8]                    # ANZECC (2000) default range for NZ lowland rivers, as in 02
CLASSES = [                                  # the four classifications, as defined in 01
    {'name': 'Critical anomaly', 'key': 'crit', 'test': 'The value is physically impossible.',
     'action': 'Remove the record; it cannot enter any analysis.'},
    {'name': 'Data-integrity error', 'key': 'int',
     'test': 'The record is provably not an independent observation: a copy of a record above (verbatim or '
             'altered), or a date that does not exist.',
     'action': 'Remove it; where it copies a record, keep the source.'},
    {'name': 'Statistical outlier', 'key': 'out',
     'test': 'Extreme by IQR or z-score, physically possible, and not provably an error.',
     'action': 'Keep, unless the record is suspect on other grounds; either way, test the alternative in the '
               'sensitivity analysis.'},
    {'name': 'Missing data', 'key': 'miss', 'test': 'A blank cell.',
     'action': 'Recover the value from a copy if one exists; infer it only if a single value fits, and flag it; '
               'otherwise leave it blank (no imputation).'},
]
CLAIMS = {                                   # the hypotheses as stated in 03, before testing
    'H1': 'Water temperature differs between sites',
    'H2': 'Dissolved oxygen differs between sites',
    'H3': 'Īnanga counts per survey differ between months',
    'H4': 'Īnanga counts are associated with oxygen saturation',
    'H5': 'Longfin eel counts are associated with oxygen saturation',
    'H6': 'Shortfin eel counts are associated with oxygen saturation',
}
SCENARIOS = {                                # sensitivity-analysis columns in 03 -> labels on the page
    'chosen': 'As cleaned',
    'same-day: first': 'Same-day readings: first, not mean',
    'row 64 = 12 Dec': 'Row 64 re-dated to 12 Dec',
    'keep row 94': 'Row 94 kept (DO 12.9 mg/L)',
    'drop row 45': 'Row 45 dropped, not inferred',
    '64 + 94 + 45': 'All three together',
}
ALTERNATIVE = {'Remove': 'If kept', 'Infer': 'If dropped'}   # first word of a matrix decision -> its alternative
_ARROW = re.compile(r'(?P<metric>.*?)\s*(?P<before>\d[\d.]*)\s*→\s*(?P<after>\d[\d.]*)\s*(?P<unit>°C|mg/L|cm)?')


def _month_name(month, fmt='%B'):
    return pd.Timestamp(f'{month}-01').strftime(fmt)


def _plain(o):
    """numpy scalars to Python and NaN to None, recursively, so the JSON is strict."""
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if hasattr(o, 'item'):
        o = o.item()
    if isinstance(o, float) and o != o:
        return None
    return o


def impact_numbers(impact, decision):
    """The first 'before → after' pair of a matrix impact sentence, with the alternative it describes."""
    m = _ARROW.match(impact)
    alternative = ALTERNATIVE.get(decision.split()[0].rstrip(':;'))
    if not m or not alternative:
        return None
    metric = re.sub(r'^[^:]*:\s*', '', m['metric'])       # 'Dropped instead: īnanga records' -> 'īnanga records'
    return {'metric': metric[:1].upper() + metric[1:], 'before': m['before'], 'after': m['after'],
            'unit': m['unit'] or '', 'alternative': alternative,
            'more': bool(impact[m.end():].strip(' ;,.'))}  # does the sentence say more than the pair?


def _load():
    visits = pd.read_csv(PROCESSED / 'visits.csv', parse_dates=['date'])
    records = pd.read_csv(PROCESSED / 'fish_records.csv', parse_dates=['date'])
    for t in (visits, records):
        t['month'] = t['date'].dt.strftime('%Y-%m')
    return visits, records


def _table(name):
    return pd.read_csv(TABLES / f'{name}.csv')


def _river(visits):
    below7 = visits.do_mg_l < 7
    summer = visits.date >= pd.Timestamp(SUMMER_STARTS)
    months = sorted(visits.month.unique())
    month_temp = visits.groupby('month').temperature_c.mean()
    medians = visits.groupby(['month', 'site_id']).temperature_c.median().unstack()
    warmest = medians.idxmax(axis=1)
    sites = _table('02_site_summary')
    ph = visits.ph.dropna()
    lo, hi = PH_GUIDELINE
    below7_sites = [s for s in SITES if (below7 & (visits.site_id == s)).any()]

    site_rows = []
    for s in sites.itertuples():
        here = visits.site_id == s.site_id
        site_rows.append({'site': s.site_id, 'visits': int(s.visits), 'temp_median': s.temp_median,
                          'temp_max': s.temp_max, 'do_min': s.do_min, 'sat_min': round(s.sat_min),
                          'sat_median': round(s.sat_median), 'below7': int(s.visits_below_7),
                          'below7_summer': int((below7 & summer & here).sum()),
                          'ph_min': visits.loc[here, 'ph'].min(), 'ph_max': visits.loc[here, 'ph'].max()})

    # Sites whose low readings all came before the summer window opened, and the sites inside it.
    notes = []
    for r in site_rows:
        if r['below7'] and not r['below7_summer']:
            low_months = visits.loc[below7 & (visits.site_id == r['site']), 'month'].unique()
            when = f'in {_month_name(low_months[0])}' if len(low_months) == 1 else 'before 1 November'
            notes.append(f"{r['site']}'s {r['below7']} readings below 7 mg/L were all {when}, "
                         'before the NPS-FM summer window opened on 1 November.')
    inside = [r for r in site_rows if r['below7_summer']]
    if notes and inside:
        notes.append('From then on, only ' + ' and '.join(f"{r['site']} ({r['below7_summer']})" for r in inside)
                     + ' fell below 7 mg/L.')
    window_note = ' '.join(notes)

    first, last = months[0], months[-1]
    return {
        'visits': len(visits),
        'below7': int(below7.sum()),
        'below_sat': int((visits.pct_sat < 100).sum()),
        'sat_min': round(visits.pct_sat.min()), 'sat_max': round(visits.pct_sat.max()),
        'month_temp': [{'month': m, 'mean': round(t, 1)} for m, t in month_temp.items()],
        'warming': round(month_temp.iloc[-1] - month_temp.iloc[0], 1),
        'warmest_every_month': warmest.nunique() == 1,
        'warmest': warmest.iloc[0],
        'sites': site_rows,
        'grid': _table('02_river_check_grid').rename(columns={'site_id': 'site'}).to_dict('records'),
        'window_note': window_note,
        'titles': {
            'do': ('Every site fell below 7 mg/L at least once' if len(below7_sites) == len(SITES)
                   else f'{len(below7_sites)} of {len(SITES)} sites fell below 7 mg/L'),
            'sat': (f'All {len(visits)} visits were below full saturation' if (visits.pct_sat < 100).all()
                    else f'{int((visits.pct_sat < 100).sum())} of {len(visits)} visits were below full saturation'),
            'temp': (f'The river warmed from {month_temp.iloc[0]:.1f} °C in {_month_name(first)} to '
                     f'{month_temp.iloc[-1]:.1f} °C in {_month_name(last)}'
                     + (f'; {warmest.iloc[0]} was warmest every month' if warmest.nunique() == 1 else '')),
            'ph': (f'pH ran {ph.min():.2f}–{ph.max():.2f}: {int((ph < lo).sum())} readings below {lo} and '
                   f'{int((ph > hi).sum())} above {hi}'),
        },
    }


def _fish(records):
    months = sorted(records.month.unique())
    summary = {}
    for site in ['all'] + SITES:
        r = records if site == 'all' else records[records.site_id == site]
        rows = []
        for sp in SPECIES:
            for m in months:
                d = r[(r.species == sp) & (r.month == m)]
                rows.append({'species': sp, 'month': m, 'n': len(d), 'total': int(d['count'].sum()),
                             'mean': round(d['count'].mean(), 1) if len(d) else None,
                             'size': round(d['size_cm'].mean(), 1) if d['size_cm'].notna().any() else None})
        summary[site] = rows

    # The title states the finding only while it holds, with the same checks as Figure D in 02.
    per = records.groupby(['species', 'month'])['count'].mean().unstack()
    ina = per.loc['Inanga']
    halved = ina.iloc[-1] < 0.5 * ina.iloc[:-1].mean()
    rose = all(per.loc[sp].iloc[-1] > per.loc[sp].iloc[:-1].max() for sp in ('Brown Trout', 'Longfin Eel'))
    title = (f'Īnanga counts per survey halved in {_month_name(months[-1])}' if halved
             else 'Īnanga counts per survey by month')
    if halved and rose:
        title += ', while trout and longfin eel counts rose'

    species = []
    for sp, name in SPECIES.items():
        d = records[records.species == sp]
        species.append({'code': sp, 'name': name, 'origin': d.origin.iloc[0], 'status': d.nz_status.iloc[0],
                        'records': len(d), 'size': round(d.size_cm.mean(), 1),
                        'inferred': int(d.species_inferred.sum())})
    return {'records': len(records), 'native': int(sum(s['origin'] == 'native' for s in species)),
            'species': species, 'summary': summary,
            'inanga': [{'month': m, 'mean': round(v, 1)} for m, v in ina.items()],
            'titles': {'all': title, **{s: f'Fish counted per survey at {s}' for s in SITES}}}


def _validation():
    log = _table('02_cleaning_log')
    flow, raw, removed, merged = [], 0, 0, 0
    for table, steps in log.groupby('table', sort=False):
        before, after = int(steps['rows before'].iloc[0]), int(steps['rows after'].iloc[-1])
        parts = []
        for _, s in steps.iterrows():
            n = int(s['rows before'] - s['rows after'])
            if n:
                parts.append({'step': s['step'], 'n': n, 'rows': str(s['Excel rows'])})
                if s['step'].startswith('drop'):
                    removed += n
                else:
                    merged += n
        raw += before
        flow.append({'table': table, 'before': before, 'after': after, 'parts': parts})

    matrix = _table('02_validation_matrix')
    keys = {c['name']: c['key'] for c in CLASSES}
    entries = [{'problem': r['Value / Problem'], 'class': r['Classification'], 'key': keys[r['Classification']],
                'decision': r['Decision'], 'impact': r['Potential ecological & statistical impact'],
                'numbers': impact_numbers(r['Potential ecological & statistical impact'], r['Decision'])}
               for _, r in matrix.iterrows()]
    return {'raw': raw, 'removed': removed, 'merged': merged, 'flow': flow, 'classes': CLASSES,
            'entries': entries,
            'title': f'{removed} of {raw} raw rows were copies, altered copies or invalid records'}


def _findings():
    tests = _table('03_hypothesis_tests')
    rows = [{'id': t['id'], 'claim': CLAIMS[t['id']], 'test': t['statistic'], 'value': t['value'],
             'effect_name': t['effect size'], 'effect': t['effect'], 'n': int(t['n']), 'p': t['p'],
             'p_holm': t['p (Holm)'], 'supported': bool(t['reject H0 at 0.05'])} for _, t in tests.iterrows()]
    perm = _table('03_h3_permutation').iloc[0]
    sens = _table('03_sensitivity')
    alternatives = [c for c in sens.columns if c not in ('result', 'chosen')]
    same = sens.set_index('result').loc['same conclusions as chosen', alternatives].astype(str)
    changed = int((same != 'True').sum())
    models = _table('03_model_vs_baseline')
    supported = sum(r['supported'] for r in rows)
    return {
        'tests': rows,
        'permutation': {'p': perm['permutation p'], 'shuffles': int(perm['shuffles']),
                        'adjusted': perm['adjusted p'], 'factor': int(perm['Holm factor'])},
        'scenarios': [{'key': c, 'label': SCENARIOS[c]} for c in ['chosen'] + alternatives],
        'sensitivity': [{'result': r['result'], 'values': [str(r[c]) for c in ['chosen'] + alternatives]}
                        for _, r in sens.iterrows()],
        'changed': changed, 'alternatives': len(alternatives),
        'models': [{'species': SPECIES[m['species']], 'n': int(m['n']), 'model': round(m['LOOCV RMSE (linear)'], 2),
                    'baseline': round(m['LOOCV RMSE (mean baseline)'], 2), 'beats': bool(m['beats baseline'])}
                   for _, m in models.iterrows()],
        'supported': supported,
        'title': (f'{supported} of {len(rows)} hypotheses are supported, and no cleaning judgement changes that'
                  if changed == 0 else f'{supported} of {len(rows)} hypotheses are supported'),
    }


def dashboard_data():
    """Everything the page shows, computed from the processed data and the results tables."""
    visits, records = _load()
    v_rows = [{'site': r.site_id, 'date': r.date.strftime('%Y-%m-%d'), 'temp': r.temperature_c,
               'ph': r.ph, 'do': r.do_mg_l, 'sat': round(r.pct_sat, 1), 'band': r.band,
               'n': int(r.n_readings), 'rows': str(r.excel_rows)} for r in visits.itertuples()]
    start, end = visits.date.min(), visits.date.max()
    return _plain({
        'meta': {
            'start': f'{start:%Y-%m}-01', 'end': (end + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d'),
            'first': start.strftime('%Y-%m-%d'), 'last': end.strftime('%Y-%m-%d'),
            'months': sorted(visits.month.unique()), 'sites': SITES, 'summer_starts': SUMMER_STARTS,
            'bands': BANDS, 'ph_guideline': PH_GUIDELINE,
            'byline': 'Eric Gomez · October 2026',
        },
        'visits': v_rows,
        'river': _river(visits),
        'fish': _fish(records),
        'validation': _validation(),
        'findings': _findings(),
    })


def render_html():
    """The page as one string: the template with the data embedded."""
    payload = json.dumps(dashboard_data(), ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    payload = payload.replace('</', '<\\/')              # no string in the data can close the <script>
    return TEMPLATE.read_text(encoding='utf-8').replace('__DATA__', payload)


def build_dashboard_html(name=NAME):
    """Write the dashboard to outputs/dashboards/<name>.html and return its path."""
    DASHBOARDS.mkdir(parents=True, exist_ok=True)
    path = DASHBOARDS / f'{name}.html'
    path.write_text(render_html(), encoding='utf-8')
    print(f'saved outputs/dashboards/{path.name} ({path.stat().st_size / 1024:.0f} KB)')
    return path


def screenshot_dashboard(path=None, views=('river',), size=(1280, 1050), scale=2):
    """PNG of each view in `views` with headless Chrome, for the README and the report. Needs a local Chrome.

    At 1280 x 1050 the River check and Fish views fit whole; the other two scroll."""
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
        screenshot_dashboard(built)
