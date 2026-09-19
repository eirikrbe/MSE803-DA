"""Tests for src/models/classification.py.

Synthetic data with a known ground truth, never the project's own dataset --
that is what keeps this file portable when the template is copied.

    data_env/bin/python3 tests/test_classification.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.classification import (compare_kernels, cv_summary,  # noqa: E402
                                       majority_class_baseline, report_baseline_gain)


# --------------------------------------------------------------------------
# majority_class_baseline
# --------------------------------------------------------------------------

def test_majority_class_baseline_picks_the_most_frequent_training_class():
    y_train = ['a'] * 7 + ['b'] * 3
    y_test = ['a'] * 4 + ['b'] * 1
    accuracy, majority_class = majority_class_baseline(y_train, y_test)
    assert majority_class == 'a', majority_class
    assert abs(accuracy - 0.8) < 1e-9, accuracy


def test_majority_class_baseline_scores_zero_when_test_is_all_the_other_class():
    y_train = ['a'] * 9 + ['b'] * 1
    y_test = ['b'] * 5
    accuracy, majority_class = majority_class_baseline(y_train, y_test)
    assert majority_class == 'a', majority_class
    assert accuracy == 0.0, accuracy


def test_majority_class_baseline_ignores_test_labels_when_choosing_the_majority():
    """The majority class must come from y_train only -- using y_test to pick
    it would leak information a real predictor would not have at test time."""
    y_train = ['a'] * 6 + ['b'] * 4
    y_test = ['b'] * 10
    accuracy, majority_class = majority_class_baseline(y_train, y_test)
    assert majority_class == 'a', 'majority class must come from train, not test'
    assert accuracy == 0.0, accuracy


def test_majority_class_baseline_accepts_pandas_series_with_a_nondefault_index():
    """train_test_split leaves the original DataFrame index on y_train/y_test;
    the function must not silently misalign on that."""
    y_train = pd.Series(['x', 'x', 'y'], index=[10, 11, 12])
    y_test = pd.Series(['x', 'y', 'x'], index=[50, 51, 52])
    accuracy, majority_class = majority_class_baseline(y_train, y_test)
    assert majority_class == 'x', majority_class
    assert abs(accuracy - 2 / 3) < 1e-9, accuracy


def test_majority_class_baseline_handles_three_classes():
    y_train = ['setosa'] * 5 + ['versicolor'] * 4 + ['virginica'] * 3
    y_test = ['setosa', 'setosa', 'versicolor', 'virginica']
    accuracy, majority_class = majority_class_baseline(y_train, y_test)
    assert majority_class == 'setosa', majority_class
    assert abs(accuracy - 0.5) < 1e-9, accuracy


# --------------------------------------------------------------------------
# report_baseline_gain
# --------------------------------------------------------------------------

def test_report_baseline_gain_computes_the_correct_positive_gain():
    gain = report_baseline_gain(0.5, 'setosa', 0.9, model_label='svm', n_test=30)
    assert abs(gain - 0.4) < 1e-9, gain


def test_report_baseline_gain_can_be_negative():
    gain = report_baseline_gain(0.6, 'setosa', 0.4, model_label='svm')
    assert gain < 0, gain
    assert abs(gain - (-0.2)) < 1e-9, gain


# --------------------------------------------------------------------------
# cv_summary
# --------------------------------------------------------------------------

def test_cv_summary_computes_mean_and_std():
    result = cv_summary([0.6, 0.8])
    assert abs(result['mean'] - 0.7) < 1e-9, result['mean']
    assert abs(result['std'] - 0.1) < 1e-9, result['std']
    assert list(result['scores']) == [0.6, 0.8]


def test_cv_summary_zero_variance_when_every_fold_agrees():
    result = cv_summary([1.0, 1.0, 1.0])
    assert result['mean'] == 1.0, result['mean']
    assert result['std'] == 0.0, result['std']


# --------------------------------------------------------------------------
# compare_kernels
# --------------------------------------------------------------------------

def _synthetic_classification(n_per_class=30, n_features=4, class_sep=3.0, seed=0):
    """Three well-separated Gaussian blobs in n_features dimensions -- known
    ground truth (which blob a row was drawn from), never the project's own
    dataset, so this file stays portable when the template is copied.

    Returns a deterministic 80/20 split (stratified by construction: each
    class is a contiguous, equal-sized block, and the same fraction of each
    block goes to test).
    """
    rng = np.random.default_rng(seed)
    classes = ['a', 'b', 'c']
    X_parts, y_parts = [], []
    for i, cls in enumerate(classes):
        center = np.zeros(n_features)
        center[i % n_features] = i * class_sep
        X_parts.append(rng.normal(loc=center, scale=0.5, size=(n_per_class, n_features)))
        y_parts.append([cls] * n_per_class)
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    n_test_per_class = max(1, n_per_class // 5)
    train_idx, test_idx = [], []
    for i in range(len(classes)):
        block = np.arange(i * n_per_class, (i + 1) * n_per_class)
        test_idx.extend(block[:n_test_per_class])
        train_idx.extend(block[n_test_per_class:])
    train_idx, test_idx = np.array(train_idx), np.array(test_idx)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = pd.Series(y[train_idx]), pd.Series(y[test_idx])
    return X_train, X_test, y_train, y_test


def test_compare_kernels_returns_one_result_per_kernel():
    X_train, X_test, y_train, y_test = _synthetic_classification()
    res = compare_kernels(X_train, y_train, X_test, y_test,
                          kernels=('linear', 'poly', 'rbf'), cv_folds=3, random_state=0)
    assert [r['kernel'] for r in res['results']] == ['linear', 'poly', 'rbf']


def test_compare_kernels_never_names_a_winner():
    """The whole point, same as compare_models on the regression side: it
    reports every kernel, you choose."""
    X_train, X_test, y_train, y_test = _synthetic_classification()
    res = compare_kernels(X_train, y_train, X_test, y_test, cv_folds=3, random_state=0)
    for forbidden in ('best', 'best_kernel', 'winner', 'recommended', 'selected'):
        assert forbidden not in res, f'compare_kernels returned {forbidden!r}'
    for r in res['results']:
        for forbidden in ('best', 'winner', 'recommended', 'selected'):
            assert forbidden not in r, f'result dict returned {forbidden!r}'


def test_compare_kernels_metrics_are_valid_accuracies():
    X_train, X_test, y_train, y_test = _synthetic_classification()
    res = compare_kernels(X_train, y_train, X_test, y_test, cv_folds=3, random_state=0)
    assert 0.0 <= res['baseline_accuracy'] <= 1.0
    for r in res['results']:
        assert 0.0 <= r['test_accuracy'] <= 1.0
        assert 0.0 <= r['cv_mean'] <= 1.0
        assert 0.0 <= r['cv_std'] <= 1.0
        assert all(0.0 <= s <= 1.0 for s in r['cv_scores'])


def test_compare_kernels_baseline_computed_from_y_train_only():
    """Same leakage guard as majority_class_baseline itself: build a y_train
    with a clear, forced majority class and confirm the reported baseline
    tracks that -- not the (differently-balanced) y_test."""
    X_train, X_test, y_train, y_test = _synthetic_classification()
    n = len(y_train)
    # At least cv_folds=3 of each minority label, so StratifiedKFold can still
    # stratify -- this test is about the baseline, not about CV correctness.
    y_train_skewed = pd.Series(['a'] * (n - 6) + ['b'] * 3 + ['c'] * 3)
    res = compare_kernels(X_train, y_train_skewed, X_test, y_test, cv_folds=3, random_state=0)
    expected_accuracy, expected_majority = majority_class_baseline(y_train_skewed, y_test)
    assert res['majority_class'] == expected_majority == 'a'
    assert abs(res['baseline_accuracy'] - expected_accuracy) < 1e-9


def test_compare_kernels_poly_degree_is_explicit_and_labelled():
    X_train, X_test, y_train, y_test = _synthetic_classification()
    res = compare_kernels(X_train, y_train, X_test, y_test, cv_folds=3, random_state=0,
                          poly_degree=2)
    poly_result = next(r for r in res['results'] if r['kernel'] == 'poly')
    assert poly_result['pipeline'].named_steps['svm'].degree == 2
    assert '2' in poly_result['label']


def test_compare_kernels_fitted_pipelines_predict_on_test_set():
    """The returned pipeline/y_pred should be directly usable (e.g. for a
    confusion matrix) without the caller needing to refit."""
    X_train, X_test, y_train, y_test = _synthetic_classification()
    res = compare_kernels(X_train, y_train, X_test, y_test, cv_folds=3, random_state=0)
    for r in res['results']:
        assert len(r['y_pred']) == len(y_test)
        refit_pred = r['pipeline'].predict(X_test)
        assert list(refit_pred) == list(r['y_pred'])


def test_compare_kernels_separable_data_beats_the_baseline():
    """Sanity check on well-separated synthetic blobs: every kernel should
    comfortably beat the majority-class baseline -- the same gain-over-
    baseline property compare_models enforces on the regression side."""
    X_train, X_test, y_train, y_test = _synthetic_classification(class_sep=5.0)
    res = compare_kernels(X_train, y_train, X_test, y_test, cv_folds=3, random_state=0)
    for r in res['results']:
        assert r['test_accuracy'] > res['baseline_accuracy'], r


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
