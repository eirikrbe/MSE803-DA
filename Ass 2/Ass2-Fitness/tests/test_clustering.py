"""Tests for src/models/clustering.py on synthetic data with known structure.

Run standalone:  data_env/bin/python3 tests/test_clustering.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from src.models.clustering import bootstrap_stability, hopkins, kmeans_sweep


def _blobs(seed=0, n=150):
    rng = np.random.default_rng(seed)
    centres = np.array([[0, 0], [10, 0], [5, 9]])
    return np.vstack([c + rng.normal(scale=0.6, size=(n, 2)) for c in centres])


def test_hopkins_uniform_near_half_and_blobs_high():
    rng = np.random.default_rng(3)
    uniform = rng.uniform(size=(1500, 3))
    h_u = hopkins(uniform, seed=0)
    assert 0.40 < h_u < 0.60, f'uniform data should give about 0.5, got {h_u:.3f}'
    h_b = hopkins(_blobs(), seed=0)
    assert h_b > 0.85, f'well-separated blobs should give a high value, got {h_b:.3f}'


def test_sweep_finds_three_blobs_and_returns_no_choice():
    tab = kmeans_sweep(_blobs(), ks=[2, 3, 4, 5], seed=0)
    assert tab.loc[tab['silhouette'].idxmax(), 'k'] == 3
    assert tab.loc[tab['davies_bouldin'].idxmin(), 'k'] == 3
    assert set(tab.columns) == {'k', 'inertia', 'silhouette', 'davies_bouldin', 'calinski_harabasz'}


def test_stability_high_for_real_clusters():
    mean, sd, draws = bootstrap_stability(_blobs(), k=3, B=8, seed=0)
    assert mean > 0.95 and len(draws) == 8


def test_hopkins_rejects_oversized_sample():
    try:
        hopkins(np.zeros((10, 2)), m=10)
    except ValueError:
        return
    raise AssertionError('m >= n must raise')


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
