"""Tests for src/ethics/privacy.py and src/ethics/fairness.py on small frames
whose answers can be checked by hand.

Run standalone:  data_env/bin/python3 tests/test_ethics.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.ethics.fairness import disparity_summary, group_metric_table
from src.ethics.privacy import (
    dp_error, equivalence_class_sizes, k_anonymity_report, laplace_mean)


def _people():
    return pd.DataFrame({
        'gender': ['F', 'F', 'F', 'M', 'M', 'M'],
        'band':   ['a', 'a', 'b', 'a', 'a', 'a'],
        'value':  [1, 2, 3, 4, 5, 6],
    })


def test_equivalence_classes_and_k():
    df = _people()
    sizes = equivalence_class_sizes(df, ['gender', 'band'])
    assert sizes.tolist() == [2, 2, 1, 3, 3, 3]
    rep = k_anonymity_report(df, {'g+b': ['gender', 'band'], 'g': ['gender']}, k=3)
    r = rep.set_index('quasi_identifiers')
    assert r.loc['g+b', 'min_k'] == 1 and r.loc['g+b', 'classes'] == 3
    assert abs(r.loc['g+b', 'pct_unique'] - 100 / 6) < 1e-9
    assert r.loc['g', 'min_k'] == 3 and r.loc['g', 'pct_below_k3'] == 0


def test_unknown_column_raises():
    try:
        equivalence_class_sizes(_people(), ['nope'])
    except KeyError:
        return
    raise AssertionError('an unknown quasi-identifier must raise')


def test_laplace_mean_clips_and_converges():
    rng = np.random.default_rng(0)
    vals = np.array([0, 10, 20, 1000])            # 1000 is clipped to 30
    est = laplace_mean(vals, 0, 30, epsilon=1e9, rng=rng)
    assert abs(est - 15) < 1e-3, 'huge epsilon must return the clipped mean'
    try:
        laplace_mean(vals, 0, 30, epsilon=0, rng=rng)
    except ValueError:
        pass
    else:
        raise AssertionError('epsilon <= 0 must raise')


def test_dp_error_falls_as_epsilon_rises():
    vals = np.random.default_rng(1).uniform(0, 100, 500)
    tab = dp_error(vals, 0, 100, epsilons=[0.1, 1, 10], B=400, seed=0)
    errs = tab['mean_abs_error'].tolist()
    assert errs[0] > errs[1] > errs[2]
    # expected |Laplace| = b = (range / n) / eps = 0.2 / eps
    assert abs(errs[1] - 0.2) < 0.05


def test_group_metric_table_and_disparity():
    df = pd.DataFrame({'grp': ['a'] * 50 + ['b'] * 50,
                       'y': [1] * 100,
                       'p': [1] * 50 + [1] * 40 + [0] * 10})
    acc = lambda t, p: float(np.mean(t == p))
    tab = group_metric_table(df, 'grp', 'y', 'p', acc, 'accuracy', B=300)
    t = tab.set_index('group')
    assert t.loc['a', 'accuracy'] == 1.0 and t.loc['b', 'accuracy'] == 0.8
    assert t.loc['a', 'n'] == 50
    s = disparity_summary(tab, 'accuracy')
    assert abs(s['gap'] - 0.2) < 1e-12 and s['groups'] == 2
    assert s['all_intervals_overlap'] is False


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
