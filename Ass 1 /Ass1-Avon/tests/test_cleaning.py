"""Tests for src/data/cleaning.py.

Runs with pytest if available, or standalone:
    data_env/bin/python3 tests/test_cleaning.py

pytest is not installed in data_env, so these are plain functions with plain
asserts -- pytest collects them, and the runner at the bottom works without it.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cleaning import (clean_number, coerce_types,  # noqa: E402
                               drop_exact_duplicates, snake_case_columns,
                               standardize_category, strip_whitespace,
                               text_to_number)


# --------------------------------------------------------------------------
# clean_number - the parser that had a real bug in W3
# --------------------------------------------------------------------------

def test_parses_plain_digits():
    assert clean_number(38) == 38.0
    assert clean_number('38') == 38.0
    assert clean_number('38.5') == 38.5


def test_parses_thousands_separators_and_symbols():
    assert clean_number('30,000') == 30000.0
    assert clean_number('$1,200') == 1200.0
    assert clean_number('45%') == 45.0


def test_hyphenated_and_spaced_forms_agree():
    assert clean_number('thirty-eight') == clean_number('thirty eight') == 38.0


def test_generalises_beyond_the_values_in_one_dataset():
    """Regression test for W3's per-column lookup bug.

    The original returned NaN for any spelled-out value it had not been told
    about. One parser must handle words it has never seen in this dataset.
    """
    for text, expected in [('forty two', 42), ('ninety-nine', 99),
                           ('seventeen', 17), ('sixty-five thousand', 65000)]:
        assert clean_number(text) == float(expected), f'{text!r} -> {expected}'


def test_blank_and_missing_become_nan():
    for bad in [None, np.nan, '', '   ', 'not a number', 'N/A']:
        assert pd.isna(clean_number(bad)), f'{bad!r} should be NaN'


def test_text_to_number_returns_none_when_nothing_recognised():
    assert text_to_number('hello world') is None
    assert text_to_number('twenty') == 20


# --------------------------------------------------------------------------
# standardize_category
# --------------------------------------------------------------------------

def test_aliases_collapse_onto_one_label():
    aliases = {'AU': 'AUS', 'AUSTRALIA': 'AUS'}
    for variant in ['AU', 'au', 'Australia', ' AUS ']:
        got = standardize_category(variant, aliases)
        assert got == 'AUS', f'{variant!r} -> {got!r}'


def test_unmapped_values_pass_through_unchanged():
    """Nothing is silently reclassified just because it was not in the map."""
    assert standardize_category('NZ', {'AU': 'AUS'}) == 'NZ'


def test_missing_category_is_nan():
    assert pd.isna(standardize_category(None, {}))
    assert pd.isna(standardize_category('  ', {}))


# --------------------------------------------------------------------------
# frame-level helpers
# --------------------------------------------------------------------------

def test_snake_case_handles_spaces_camel_and_symbols():
    df = pd.DataFrame(columns=['Happiness Score', 'GDPPerCapita', 'Net-Worth ($)', 'ok'])
    got = list(snake_case_columns(df).columns)
    assert got == ['happiness_score', 'gdp_per_capita', 'net_worth', 'ok'], got


def test_snake_case_does_not_mutate_the_input():
    df = pd.DataFrame(columns=['A Column'])
    snake_case_columns(df)
    assert list(df.columns) == ['A Column'], 'input frame was mutated'


def test_strip_whitespace_leaves_non_strings_alone():
    df = pd.DataFrame({'a': ['  x  ', 'y'], 'n': [1, 2]})
    out = strip_whitespace(df)
    assert list(out['a']) == ['x', 'y']
    assert list(out['n']) == [1, 2]


def test_coerce_types_routes_numeric_through_clean_number():
    df = pd.DataFrame({'salary': ['30,000', 'sixty-five thousand', '']})
    out = coerce_types(df, {'salary': 'numeric'})
    assert out['salary'].iloc[0] == 30000.0
    assert out['salary'].iloc[1] == 65000.0
    assert pd.isna(out['salary'].iloc[2])


def test_coerce_types_rejects_an_unknown_column():
    df = pd.DataFrame({'a': [1]})
    try:
        coerce_types(df, {'nope': 'numeric'})
    except KeyError:
        return
    raise AssertionError('expected KeyError for a column that does not exist')


def test_drop_exact_duplicates_reports_and_removes():
    df = pd.DataFrame({'a': [1, 1, 2], 'b': ['x', 'x', 'y']})
    out = drop_exact_duplicates(df)
    assert len(out) == 2, f'expected 2 rows, got {len(out)}'
    assert len(df) == 3, 'input frame was mutated'


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
