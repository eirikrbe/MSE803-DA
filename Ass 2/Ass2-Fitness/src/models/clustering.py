"""Clustering diagnostics: is there structure to find, and is what k-means
finds stable?

k-means always returns k groups, whether or not the data contain any. So the
diagnostics come in three parts, and none of them chooses k for you:

1. Clustering tendency (`hopkins`): does the data depart from uniform at all?
2. Internal quality across k (`kmeans_sweep`): inertia, silhouette,
   Davies-Bouldin and Calinski-Harabasz side by side.
3. Stability (`bootstrap_stability`): would a resample give the same partition?

A partition can be stable and still arbitrary: k-means cuts a uniform cloud the
same way every time. Stability is necessary for a segment to be real, not
sufficient.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors


def hopkins(X, m=None, seed=0):
    """Hopkins statistic (Hopkins & Skellam, 1954; Lawson & Jurs, 1990 form).

    Compares nearest-neighbour distances of m real points with those of m points
    drawn uniformly in the data's bounding box. About 0.5 means the data look
    uniform; values towards 1 mean clustered. Caveat: correlated variables leave
    most of the bounding box empty, which pushes H up even when there are no
    clusters, so run it on decorrelated features too.
    """
    X = np.asarray(X, float)
    n, d = X.shape
    m = m or max(10, n // 10)
    if m >= n:
        raise ValueError('m must be smaller than the number of rows')
    rng = np.random.default_rng(seed)
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    sample = X[rng.choice(n, m, replace=False)]
    w = nn.kneighbors(sample, n_neighbors=2)[0][:, 1]          # skip the point itself
    u_pts = rng.uniform(X.min(axis=0), X.max(axis=0), size=(m, d))
    u = nn.kneighbors(u_pts, n_neighbors=1)[0][:, 0]
    return float(u.sum() / (u.sum() + w.sum()))


def kmeans_sweep(X, ks, seed=0, n_init=10, silhouette_sample=None):
    """Fit k-means for each k in `ks` and report four internal measures.

    Higher silhouette (max 1) and Calinski-Harabasz are better; lower
    Davies-Bouldin is better; inertia always falls as k grows, so read it for an
    elbow, not a minimum. Returns one row per k and no chosen k.
    """
    X = np.asarray(X, float)
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X)
        rows.append({
            'k': k,
            'inertia': float(km.inertia_),
            'silhouette': float(silhouette_score(
                X, km.labels_, sample_size=silhouette_sample, random_state=seed)),
            'davies_bouldin': float(davies_bouldin_score(X, km.labels_)),
            'calinski_harabasz': float(calinski_harabasz_score(X, km.labels_)),
        })
    return pd.DataFrame(rows)


def bootstrap_stability(X, k, B=20, seed=0, n_init=10):
    """Mean and sd of the adjusted Rand index between the full-data partition and
    partitions refitted on bootstrap resamples (each predicted back onto all rows).

    ARI near 1: the same segments come back. Near 0: the partition is noise.
    Returns (mean, sd, draws).
    """
    X = np.asarray(X, float)
    rng = np.random.default_rng(seed)
    ref = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X).labels_
    draws = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, len(X), len(X))
        km = KMeans(n_clusters=k, n_init=n_init, random_state=seed + b + 1).fit(X[idx])
        draws[b] = adjusted_rand_score(ref, km.predict(X))
    return float(draws.mean()), float(draws.std(ddof=1)), draws
