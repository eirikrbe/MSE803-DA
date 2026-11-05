"""Subgroup performance audit: does a model work equally well for every group?

A single overall score can hide a group the model serves badly. These functions
split the test set by a group column and report the metric per group with a
bootstrap interval and the group size, so a gap can be judged against noise.

Which groups to audit, and how large a gap is acceptable, are decisions for the
analyst and the people affected; nothing here sets a threshold.
"""

import numpy as np
import pandas as pd


def group_metric_table(df, group_col, y_true_col, y_pred_col, metric,
                       metric_name='metric', B=1000, seed=0, level=0.95):
    """Per-group metric with a percentile bootstrap interval and n.

    `metric(y_true, y_pred)` is any scoring function, e.g. mean absolute error
    or accuracy. Rows are resampled within each group.
    """
    rng = np.random.default_rng(seed)
    tail = (1 - level) / 2
    rows = []
    for g, part in df.groupby(group_col, observed=True):
        yt = part[y_true_col].to_numpy()
        yp = part[y_pred_col].to_numpy()
        n = len(yt)
        draws = np.empty(B)
        for b in range(B):
            idx = rng.integers(0, n, n)
            draws[b] = metric(yt[idx], yp[idx])
        rows.append({'attribute': group_col, 'group': str(g), 'n': n,
                     metric_name: float(metric(yt, yp)),
                     'ci_low': float(np.quantile(draws, tail)),
                     'ci_high': float(np.quantile(draws, 1 - tail))})
    return pd.DataFrame(rows)


def disparity_summary(table, metric_name='metric'):
    """Largest gap between groups in one attribute, as a difference and a ratio,
    and whether every group's interval overlaps every other's.

    Overlapping intervals are a rough screen, not a test: they mean the gap is
    within what resampling alone could produce at these group sizes.
    """
    vals = table[metric_name]
    lo, hi = table['ci_low'], table['ci_high']
    overlap = bool(lo.max() <= hi.min())
    return {
        'attribute': table['attribute'].iloc[0],
        'groups': len(table),
        'min': float(vals.min()), 'max': float(vals.max()),
        'gap': float(vals.max() - vals.min()),
        'ratio': float(vals.max() / vals.min()) if vals.min() > 0 else np.inf,
        'all_intervals_overlap': overlap,
    }
