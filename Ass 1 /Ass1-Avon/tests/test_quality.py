"""Tests for src/data/quality.py.

The contract these pin down: every function REPORTS and none of them mutates or
drops. That is the property the whole "expose the judgement" design rests on, so
it is the property worth testing.

    data_env/bin/python3 tests/test_quality.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.quality import (cardinality_report, duplicate_report,  # noqa: E402
                              missing_report, outlier_report, schema_report)


def _toy():
    """A frame with one of everything: a gap, a duplicate, an outlier, a constant.

    Row 5 repeats row 0 exactly (an exact duplicate); 'd' repeats on the id column
    only (a subset duplicate). The two are different findings and the report
    distinguishes them.
    """
    return pd.DataFrame({
        'id': ['a', 'b', 'c', 'd', 'd', 'a'],
        'value': [10.0, 11.0, np.nan, 12.0, 12.0, 10.0],
        'wild': [1.0, 2.0, 3.0, 4.0, 900.0, 1.0],
        'constant': ['x'] * 6,
    })


# --------------------------------------------------------------------------
# nothing here mutates or drops - the core contract
# --------------------------------------------------------------------------

def test_no_report_mutates_or_drops():
    df = _toy()
    before = df.copy(deep=True)
    for fn in (schema_report, missing_report, outlier_report, cardinality_report):
        out = fn(df)
        assert len(df) == len(before), f'{fn.__name__} changed the row count'
        pd.testing.assert_frame_equal(df, before, obj=f'{fn.__name__} mutated input')
        assert isinstance(out, pd.DataFrame), f'{fn.__name__} should return a report'


def test_outlier_report_flags_but_keeps_every_row():
    df = _toy()
    report = outlier_report(df)
    wild = report[report['column'] == 'wild'].iloc[0]
    assert wild['iqr_flagged'] >= 1, 'the 900 should be flagged'
    assert wild['n'] == 6, 'the flagged row must still be counted, not removed'


# --------------------------------------------------------------------------
# the reports say what is actually there
# --------------------------------------------------------------------------

def test_schema_report_counts_missing_per_column():
    report = schema_report(_toy()).set_index('column')
    assert report.loc['value', 'missing'] == 1
    assert report.loc['value', 'non_null'] == 5
    assert report.loc['id', 'missing'] == 0


def test_missing_report_lists_only_columns_with_gaps():
    report = missing_report(_toy())
    assert list(report.index) == ['value'], list(report.index)
    assert report.loc['value', 'missing_pct'] == 16.67


def test_duplicate_report_counts_exact_and_subset():
    result = duplicate_report(_toy(), subset=['id'])
    assert result['exact_duplicates'] == 1, 'row 5 repeats row 0 exactly'
    assert result['duplicates_on_subset'] == 2, "'a' and 'd' each repeat on id"


def test_cardinality_flags_a_constant_column():
    report = cardinality_report(_toy()).set_index('column')
    assert 'constant' in report.loc['constant', 'note']


def test_cardinality_does_not_call_continuous_numerics_identifiers():
    """A near-unique float is a measurement, not an ID. Regression test."""
    df = pd.DataFrame({'measure': np.linspace(0, 1, 20)})
    note = cardinality_report(df).set_index('column').loc['measure', 'note']
    assert note == '', f'continuous numeric wrongly annotated: {note!r}'


def test_outlier_report_skips_columns_with_too_little_data():
    df = pd.DataFrame({'tiny': [1.0, 2.0, np.nan, np.nan]})
    assert outlier_report(df).empty, 'n < 4 should not produce fences'


def test_reports_survive_an_all_missing_column():
    df = pd.DataFrame({'empty': [np.nan] * 5, 'ok': [1.0, 2, 3, 4, 5]})
    assert schema_report(df).set_index('column').loc['empty', 'non_null'] == 0
    assert missing_report(df).loc['empty', 'missing_pct'] == 100.0


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
