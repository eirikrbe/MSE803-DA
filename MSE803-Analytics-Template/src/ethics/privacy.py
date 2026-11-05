"""Re-identification risk and privacy-preserving release.

Removing names and IDs does not make a dataset anonymous. A combination of
ordinary attributes (quasi-identifiers) can single a person out (Sweeney, 2002).
These functions measure that risk on the data itself and show what
generalisation or differential privacy costs in accuracy.

Which columns count as quasi-identifiers is a judgement about what an attacker
could know, so every function takes the list explicitly.
"""

import numpy as np
import pandas as pd


def equivalence_class_sizes(df, quasi):
    """For each row, how many rows share its exact values on `quasi`.

    A row in a class of size 1 is unique: anyone who knows those attributes
    has found that person's record.
    """
    missing = [c for c in quasi if c not in df.columns]
    if missing:
        raise KeyError(f'not in the frame: {missing}')
    return df.groupby(list(quasi), observed=True, dropna=False)[quasi[0]].transform('size')


def k_anonymity_report(df, quasi_sets, k=5):
    """One row per named quasi-identifier set: number of equivalence classes, the
    smallest class (the dataset's k), the share of rows that are unique, and the
    share in classes smaller than `k`.

    `quasi_sets` maps a label to a list of columns, e.g.
    {'demographics': ['gender', 'age', 'location']}.
    """
    rows = []
    for label, quasi in quasi_sets.items():
        sizes = equivalence_class_sizes(df, list(quasi))
        rows.append({
            'quasi_identifiers': label,
            'columns': ', '.join(quasi),
            'classes': int(df.groupby(list(quasi), observed=True, dropna=False).ngroups),
            'min_k': int(sizes.min()),
            'pct_unique': float(100 * (sizes == 1).mean()),
            f'pct_below_k{k}': float(100 * (sizes < k).mean()),
        })
    return pd.DataFrame(rows)


def laplace_mean(values, lower, upper, epsilon, rng):
    """Differentially private mean (Laplace mechanism; Dwork & Roth, 2014).

    Values are clipped to [lower, upper], so one person can move the mean by at
    most (upper - lower) / n. That is the sensitivity; the noise scale is
    sensitivity / epsilon. Smaller epsilon means stronger privacy and more noise.
    Treats n as public, which is the usual convention for a released group size.
    """
    if epsilon <= 0:
        raise ValueError('epsilon must be positive')
    x = np.clip(np.asarray(values, float), lower, upper)
    n = len(x)
    if n == 0:
        raise ValueError('no values')
    sensitivity = (upper - lower) / n
    return float(x.mean() + rng.laplace(0.0, sensitivity / epsilon))


def dp_error(values, lower, upper, epsilons, B=1000, seed=0):
    """Expected error of the Laplace mean at each epsilon, by simulation.

    Returns the mean absolute error and the 95th-percentile absolute error
    against the true (clipped) mean, per epsilon. Use it to choose an epsilon
    whose error the dashboard can tolerate.
    """
    rng = np.random.default_rng(seed)
    x = np.clip(np.asarray(values, float), lower, upper)
    truth = float(x.mean())
    rows = []
    for eps in epsilons:
        errs = np.abs(np.array([laplace_mean(x, lower, upper, eps, rng) for _ in range(B)]) - truth)
        rows.append({'epsilon': eps, 'n': len(x), 'true_mean': truth,
                     'mean_abs_error': float(errs.mean()),
                     'p95_abs_error': float(np.quantile(errs, 0.95))})
    return pd.DataFrame(rows)
