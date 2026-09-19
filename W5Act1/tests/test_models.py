"""Tests for src/models/ and src/analysis/hypothesis.py.

Two properties matter here and both are tested against synthetic data with a
known ground truth, never against a project's own dataset -- that is what keeps
this file portable when the template is copied.

  1. The maths is right (a known slope is recovered; LOOCV behaves).
  2. The judgement calls cannot fire on their own (method= and test= are required,
     and compare_models never names a winner).

    data_env/bin/python3 tests/test_models.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.hypothesis import compare_groups, permutation_test  # noqa: E402
from src.models.baseline import (build_model, compare_models, equation_text,  # noqa: E402
                                 evaluate, loocv_mean_baseline, loocv_rmse,
                                 predict_with)
from src.models.imputation import impute, impute_with_model  # noqa: E402


def _toy(n=30, slope=3.0, intercept=2.0, noise=0.05, seed=0):
    """A straight line with known coefficients and very little noise."""
    rng = np.random.default_rng(seed)
    X = np.linspace(0, 10, n).reshape(-1, 1)
    y = slope * X.ravel() + intercept + rng.normal(0, noise, n)
    return X, y


def _toy_frame(**kw):
    X, y = _toy(**kw)
    return pd.DataFrame({'x': X.ravel(), 'y': y})


# --------------------------------------------------------------------------
# the maths
# --------------------------------------------------------------------------

def test_recovers_a_known_slope_and_intercept():
    X, y = _toy()
    model, poly = build_model(X, y, degree=1)
    assert abs(model.coef_[0] - 3.0) < 0.05, f'slope {model.coef_[0]}'
    assert abs(model.intercept_ - 2.0) < 0.10, f'intercept {model.intercept_}'


def test_predictions_round_trip_through_the_polynomial_transform():
    X, y = _toy()
    for degree in (1, 2, 3):
        model, poly = build_model(X, y, degree)
        pred = predict_with(model, poly, X)
        assert pred.shape == y.shape
        assert np.corrcoef(pred, y)[0, 1] > 0.99, f'degree {degree} lost the signal'


def test_polynomial_never_has_lower_training_r2():
    """More flexibility cannot fit the training data worse. This is why train R2
    cannot rank models, and the reason LOOCV exists in this module."""
    X, y = _toy()
    r2_1 = evaluate(X, y, 1, 'x', 'y')['r2']
    r2_2 = evaluate(X, y, 2, 'x', 'y')['r2']
    assert r2_2 >= r2_1 - 1e-9, f'{r2_2} < {r2_1}'


def test_loocv_beats_the_mean_baseline_on_a_real_relationship():
    X, y = _toy()
    assert loocv_rmse(X, y, 1) < loocv_mean_baseline(y)


def test_loocv_does_not_beat_the_baseline_on_pure_noise():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 1))
    y = rng.normal(size=40)
    assert loocv_rmse(X, y, 1) >= loocv_mean_baseline(y) * 0.95


def test_equation_text_reads_back_the_fitted_line():
    X, y = _toy()
    model, _ = build_model(X, y, 1)
    text = equation_text(model, 1, 'age', 'salary')
    assert text.startswith('salary =') and 'age' in text, text


# --------------------------------------------------------------------------
# the judgement calls cannot fire on their own
# --------------------------------------------------------------------------

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    except Exception as other:
        raise AssertionError(f'expected {exc.__name__}, got {type(other).__name__}: {other}')
    raise AssertionError('expected an error, none was raised')


def test_compare_models_never_names_a_winner():
    """The whole point: it reports, you choose."""
    res = compare_models(_toy_frame(), 'x', 'y', degrees=(1, 2))
    for forbidden in ('best', 'best_model', 'winner', 'recommended', 'selected'):
        assert forbidden not in res, f'compare_models returned {forbidden!r}'
    assert 'results' in res and 'baseline_loocv' in res


def test_compare_models_refuses_too_few_rows():
    tiny = pd.DataFrame({'x': [1.0, 2.0], 'y': [1.0, 2.0]})
    assert compare_models(tiny, 'x', 'y') is None


def test_imputation_requires_an_explicit_method():
    df = pd.DataFrame({'a': [1.0, np.nan, 3.0]})
    _raises(lambda: impute(df, 'a'))
    _raises(lambda: impute(df, 'a', method='magic'))
    _raises(lambda: impute_with_model(df, 'a', predictors=['b']))


def test_imputation_error_lists_the_options():
    df = pd.DataFrame({'a': [1.0, np.nan]})
    try:
        impute(df, 'a')
    except ValueError as exc:
        assert 'mean' in str(exc) and 'linear_regression' in str(exc), str(exc)


def test_compare_groups_requires_an_explicit_test():
    df = pd.DataFrame({'v': [1.0, 2, 3, 4], 'g': ['a', 'a', 'b', 'b']})
    _raises(lambda: compare_groups(df, 'v', 'g', test=None))
    _raises(lambda: compare_groups(df, 'v', 'g', test='ttest'))


def test_compare_groups_refuses_more_than_two_levels():
    df = pd.DataFrame({'v': [1.0, 2, 3, 4, 5, 6], 'g': list('aabbcc')})
    _raises(lambda: compare_groups(df, 'v', 'g', test='welch_t'))


# --------------------------------------------------------------------------
# imputation provenance
# --------------------------------------------------------------------------

def test_imputation_records_provenance_and_flags_filled_rows():
    df = pd.DataFrame({'id': ['a', 'b', 'c'], 'v': [10.0, np.nan, 20.0]})
    out, prov = impute(df, 'v', method='mean', id_col='id', confidence='low')

    assert out['v'].isna().sum() == 0, 'the gap should be filled'
    assert list(out['v_was_imputed']) == [False, True, False]
    assert len(prov) == 1
    assert prov.iloc[0]['method'] == 'mean'
    assert prov.iloc[0]['confidence'] == 'low'
    assert prov.iloc[0]['id'] == 'b'
    assert df['v'].isna().sum() == 1, 'the input frame was mutated'


def test_imputation_on_a_complete_column_is_a_no_op():
    df = pd.DataFrame({'v': [1.0, 2.0]})
    out, prov = impute(df, 'v', method='mean')
    assert prov.empty
    assert 'v_was_imputed' not in out.columns


# --------------------------------------------------------------------------
# permutation test
# --------------------------------------------------------------------------

def test_permutation_test_finds_no_signal_in_noise():
    rng = np.random.default_rng(3)
    y = rng.normal(size=40)
    _, _, p = permutation_test(y, np.full(40, y.mean()), B=300, seed=1)
    assert p > 0.05, f'p = {p} on pure noise'


def test_permutation_test_detects_a_real_signal():
    X, y = _toy(n=40)
    model, poly = build_model(X, y, 1)
    _, _, p = permutation_test(y, predict_with(model, poly, X), B=300, seed=1)
    assert p < 0.05, f'p = {p} on a near-perfect fit'


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
