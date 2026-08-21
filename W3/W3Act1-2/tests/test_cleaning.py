"""Tests for the W3Act1-2 cleaning and modelling helpers.

Runs with pytest if available, or standalone:
    data_env/bin/python3 W3/W3Act1-2/tests/test_cleaning.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cleaning import clean_number, standardize_country, text_to_number  # noqa: E402
from src.modeling import (build_model, evaluate, loocv_mean_baseline,  # noqa: E402
                          loocv_rmse, permutation_test, predict_with)


# --------------------------------------------------------------------------
# clean_number - the parser that had a real bug
# --------------------------------------------------------------------------

def test_parses_plain_digits():
    assert clean_number('38') == 38.0
    assert clean_number(38) == 38.0


def test_strips_thousands_separators():
    assert clean_number('30,000') == 30000.0


def test_parses_spelled_out_numbers():
    assert clean_number('thirty-eight') == 38.0
    assert clean_number('forty two') == 42.0
    assert clean_number('sixty five thousand') == 65000.0


def test_hyphenated_and_spaced_forms_agree():
    """The original bug: hyphens were never handled, so one form silently became NaN."""
    assert clean_number('sixty-five thousand') == clean_number('sixty five thousand')
    assert clean_number('thirty-eight') == clean_number('thirty eight')


def test_generalises_beyond_the_values_in_this_dataset():
    """Regression test: clean_age() used to hardcode {'thirty-eight': 38} and
    returned NaN for every other spelled-out age."""
    for text, expected in [('twenty-five', 25.0), ('forty-two', 42.0), ('ninety', 90.0)]:
        assert clean_number(text) == expected, f'{text} should parse to {expected}'


def test_missing_and_unparseable_become_nan():
    for bad in ['', '   ', None, np.nan, 'unknown', 'n/a']:
        assert pd.isna(clean_number(bad)), f'{bad!r} should be NaN'


def test_text_to_number_returns_none_when_nothing_recognised():
    assert text_to_number('unknown') is None
    assert text_to_number('sixty five') == 65


# --------------------------------------------------------------------------
# standardize_country
# --------------------------------------------------------------------------

def test_country_variants_collapse():
    assert standardize_country('AU') == 'AUS'
    assert standardize_country('aus') == 'AUS'
    assert standardize_country(' au ') == 'AUS'
    assert standardize_country('NZ') == 'NZ'
    assert pd.isna(standardize_country(None))


# --------------------------------------------------------------------------
# modelling invariants
# --------------------------------------------------------------------------

def _toy():
    rng = np.random.default_rng(0)
    X = np.linspace(0, 10, 30).reshape(-1, 1)
    y = 3.0 * X.ravel() + 2.0 + rng.normal(0, 0.05, 30)
    return X, y


def test_linear_model_recovers_a_known_slope():
    X, y = _toy()
    model, poly = build_model(X, y, 1)
    assert poly is None
    assert abs(model.coef_[0] - 3.0) < 0.05
    assert abs(model.intercept_ - 2.0) < 0.1


def test_polynomial_adds_a_squared_term():
    X, y = _toy()
    model, poly = build_model(X, y, 2)
    assert poly is not None
    assert len(model.coef_) == 2
    assert predict_with(model, poly, X).shape == y.shape


def test_polynomial_never_has_lower_training_r2_when_well_conditioned():
    """The nested-model property the notebook relies on: adding a term cannot
    reduce training R^2 in exact arithmetic."""
    X, y = _toy()
    assert evaluate(X, y, 2, 'x', 'y')['r2'] >= evaluate(X, y, 1, 'x', 'y')['r2'] - 1e-12


def test_loocv_beats_mean_baseline_on_a_genuinely_linear_relationship():
    X, y = _toy()
    assert loocv_rmse(X, y, 1) < loocv_mean_baseline(y)


def test_permutation_test_finds_no_signal_in_noise():
    """Random targets should not clear the baseline more often than chance."""
    rng = np.random.default_rng(1)
    data = pd.DataFrame({'x': rng.normal(size=25), 'y': rng.normal(size=25)})
    _, _, p = permutation_test(data, 'x', 'y', B=200, seed=0)
    assert p > 0.05, f'noise should not look significant, got p={p}'


def test_permutation_test_detects_a_real_signal():
    X, y = _toy()
    data = pd.DataFrame({'x': X.ravel(), 'y': y})
    _, _, p = permutation_test(data, 'x', 'y', B=200, seed=0)
    assert p < 0.05, f'a clean linear relationship should be significant, got p={p}'


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
