"""Tests for src/models/evaluation.py and the effect-size helpers in
src/analysis/hypothesis.py, on synthetic data with a known answer.

Run standalone:  data_env/bin/python3 tests/test_evaluation.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from src.analysis.hypothesis import (
    cramers_v, holm_adjust, rank_biserial, spearman_ci, epsilon_squared)
from src.models.evaluation import (
    bootstrap_ci, calibration_summary, classification_metrics, cost_threshold, cv_table,
    interval_coverage, mcnemar, normalised_confusion, one_se_choice, per_class_report,
    regression_metrics, threshold_report, wilson_ci)


def test_regression_metrics_perfect_and_mean():
    y = np.array([1.0, 2, 3, 4, 5])
    m = regression_metrics(y, y)
    assert m['rmse'] == 0 and m['mae'] == 0 and m['r2'] == 1
    m = regression_metrics(y, np.full(5, y.mean()))
    assert abs(m['r2']) < 1e-12, 'predicting the mean must give R2 = 0'


def test_classification_metrics_known_case():
    y = np.array(['a', 'a', 'b', 'b'])
    p = np.array(['a', 'b', 'b', 'b'])
    m = classification_metrics(y, p, labels=['a', 'b'])
    assert m['accuracy'] == 0.75
    # class a: precision 1, recall 0.5; class b: precision 2/3, recall 1
    assert abs(m['recall_macro'] - 0.75) < 1e-12
    assert abs(m['precision_macro'] - (1 + 2 / 3) / 2) < 1e-12


def test_auc_accepts_labels_in_a_natural_not_sorted_order():
    y = np.array(['low', 'mid', 'high', 'low', 'mid', 'high'])
    # columns follow the natural order low, mid, high; perfect scores
    proba = np.array([[.9, .05, .05], [.05, .9, .05], [.05, .05, .9]] * 2)
    m = classification_metrics(y, y, proba=proba, labels=['low', 'mid', 'high'])
    assert m['roc_auc_ovr'] == 1.0
    # the same scores with two columns swapped must no longer be perfect
    m2 = classification_metrics(y, y, proba=proba[:, [1, 0, 2]], labels=['low', 'mid', 'high'])
    assert m2['roc_auc_ovr'] < 1.0


def test_per_class_and_confusion_follow_label_order():
    y = ['x', 'y', 'y', 'z']
    p = ['x', 'y', 'z', 'z']
    rep = per_class_report(y, p, labels=['z', 'y', 'x'])
    assert rep['class'].tolist() == ['z', 'y', 'x']
    cm = normalised_confusion(y, p, labels=['x', 'y', 'z'])
    assert np.allclose(cm.sum(axis=1), 1)
    assert cm.loc['y', 'y'] == 0.5 and cm.loc['y', 'z'] == 0.5


def test_bootstrap_ci_brackets_point_and_narrows_with_n():
    rng = np.random.default_rng(1)
    acc = lambda t, p: float(np.mean(t == p))
    small_t = rng.integers(0, 2, 100); small_p = np.where(rng.random(100) < 0.8, small_t, 1 - small_t)
    big_t = rng.integers(0, 2, 5000); big_p = np.where(rng.random(5000) < 0.8, big_t, 1 - big_t)
    pt, lo, hi = bootstrap_ci(small_t, small_p, acc, B=500)
    assert lo <= pt <= hi
    _, lo2, hi2 = bootstrap_ci(big_t, big_p, acc, B=500)
    assert (hi2 - lo2) < (hi - lo), 'a larger test set must give a narrower interval'


def test_cv_table_flips_negative_errors_and_names_no_winner():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 1)); y = 3 * X[:, 0] + rng.normal(scale=0.1, size=200)
    tab = cv_table({'mean': DummyRegressor(), 'linear': LinearRegression()}, X, y,
                   cv=KFold(5, shuffle=True, random_state=0),
                   scoring={'rmse': 'neg_root_mean_squared_error', 'r2': 'r2'})
    assert list(tab['model']) == ['mean', 'linear']
    assert (tab['rmse_mean'] > 0).all(), 'neg_ scores must be flipped to positive'
    assert tab.set_index('model').loc['linear', 'r2_mean'] > 0.99
    assert not any('best' in c or 'winner' in c for c in tab.columns)


def test_mcnemar_identical_and_discordant():
    y = np.array([1, 1, 1, 1, 0, 0])
    assert mcnemar(y, y, y)['p_value'] == 1.0
    a = np.array([1, 1, 1, 1, 0, 0]); b = np.array([0, 0, 0, 1, 0, 0])
    r = mcnemar(y, a, b)
    assert r['a_right_b_wrong'] == 3 and r['a_wrong_b_right'] == 0


def test_interval_coverage():
    r = interval_coverage([1, 2, 3, 10], [0, 0, 0, 0], [5, 5, 5, 5])
    assert r['coverage'] == 0.75 and r['mean_width'] == 5


def test_holm_matches_hand_calculation():
    adj = holm_adjust([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02])


def test_effect_sizes_at_their_extremes():
    assert cramers_v([[50, 50], [50, 50]]) == 0.0
    assert abs(cramers_v([[100, 0], [0, 100]]) - 1) < 1e-12
    assert rank_biserial([5, 6, 7], [1, 2, 3]) == 1.0
    assert epsilon_squared(0.0, 100) == 0.0
    lo, hi = spearman_ci(0.3, 500)
    assert lo < 0.3 < hi


def test_cv_table_train_scores_show_memorisation():
    from sklearn.tree import DecisionTreeRegressor
    rng = np.random.default_rng(1)
    X = pd.DataFrame({'x': rng.normal(size=200)})
    y = rng.normal(size=200)                      # pure noise: nothing to learn
    t = cv_table({'tree': DecisionTreeRegressor(random_state=0)}, X, y, cv=KFold(5),
                 scoring={'r2': 'r2'}, return_train=True)
    assert t['r2_train_mean'].iat[0] > 0.99 and t['r2_mean'].iat[0] < 0, 'an unpruned tree memorises noise'


def test_one_se_choice_prefers_simplest_tied_model():
    cv = pd.DataFrame({'model': ['baseline', 'simple', 'complex', 'worse'],
                       'acc_mean': [0.30, 0.775, 0.779, 0.70], 'acc_sd': [0.0, 0.01, 0.0143, 0.01]})
    chosen, cand = one_se_choice(cv, 'acc', ['simple', 'complex', 'worse'], n_folds=5)
    assert chosen == 'simple', 'within one SE (0.0064) of the best, so the simpler model wins'
    assert 'baseline' not in set(cand['model'])
    chosen, _ = one_se_choice(cv.assign(acc_sd=0.001), 'acc', ['simple', 'complex'], n_folds=5)
    assert chosen == 'complex', 'outside one SE, the best model wins'


def test_wilson_and_coverage_interval():
    lo, hi = wilson_ci(80, 100)
    assert lo < 0.8 < hi and 0.70 < lo < 0.72 and 0.86 < hi < 0.87, (lo, hi)
    lo, hi = wilson_ci(0, 10)
    assert abs(lo) < 1e-12 and 0.25 < hi < 0.35, 'Wilson stays inside [0, 1] at zero successes'
    cov = interval_coverage([1, 2, 3, 4], [0, 0, 0, 5], [5, 5, 5, 6])
    assert cov['coverage_low'] < cov['coverage'] == 0.75 < cov['coverage_high']


def test_calibration_summary_on_calibrated_probabilities():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.02, 0.98, 20000)
    y = rng.uniform(size=p.size) < p
    s, table = calibration_summary(y, p)
    assert abs(s['slope'] - 1) < 0.08 and abs(s['intercept']) < 0.08
    assert (table['observed'] - table['mean_predicted']).abs().max() < 0.03
    s_over, _ = calibration_summary(y, np.where(p > 0.5, 0.99, 0.01))
    assert s_over['slope'] < 1 and s_over['brier'] > s['brier'], 'overconfident: slope below 1'


def test_cost_threshold_lowers_as_misses_get_dearer():
    rng = np.random.default_rng(5)
    y = rng.uniform(size=2000) < 0.3
    p = np.clip(0.3 + 0.4 * (y - 0.3) + rng.normal(0, 0.2, y.size), 0, 1)
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    t1, t5 = cost_threshold(y, p, 1, grid), cost_threshold(y, p, 5, grid)
    assert t5 < t1
    r = threshold_report(y, p, t5)
    assert r['flagged'] == r['false_alarms'] + int(y.sum()) - r['missed']


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
