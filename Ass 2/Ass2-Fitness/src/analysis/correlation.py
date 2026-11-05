"""Correlation utilities.

Computing a correlation is mechanical. Choosing the method, and deciding what a
coefficient means, is not -- so `method` is explicit and every result carries the
reminder that correlation is neither causation nor predictor selection.
"""

import numpy as np


def correlation_matrix(df, method='pearson', columns=None):
    """Correlation matrix over numeric columns.

    method is explicit rather than defaulted-and-forgotten:
      'pearson'  - linear relationships, sensitive to outliers, assumes interval data
      'spearman' - monotonic relationships on ranks, robust to outliers and skew
      'kendall'  - rank-based, better for small n or many ties

    Pass the one your data justifies. If the distributions are skewed or the
    sample is small, pearson is often the wrong tool even though it is the default
    everywhere else.
    """
    numeric = df[columns] if columns is not None else df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        raise ValueError('need at least two numeric columns to correlate')
    return numeric.corr(method=method)


def top_correlations_with(df, target, n=10, method='pearson', columns=None):
    """Every numeric column's correlation with `target`, strongest absolute first.

    This ranks association. It does NOT select predictors and it does not imply
    direction: a variable can correlate strongly and be useless in a model
    (collinear with another, or a proxy for the target), and a weak correlate can
    matter once other variables are held constant. Choosing predictors is a
    modelling decision -- see the DECISION cell in 04_modeling.
    """
    numeric = df[columns] if columns is not None else df.select_dtypes(include=[np.number])
    if target not in numeric.columns:
        raise KeyError(f'{target!r} is not a numeric column. '
                       f'Have: {list(numeric.columns)}')
    corr = numeric.corr(method=method)[target].drop(target)
    out = corr.reindex(corr.abs().sort_values(ascending=False).index).head(n)

    print(f'{method} correlation with {target!r} (n = {len(numeric.dropna())} complete rows)')
    print('Association only. Not causation, and not a predictor shortlist.')
    return out.round(4)
