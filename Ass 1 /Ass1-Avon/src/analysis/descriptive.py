"""Descriptive summaries. Mechanical, and the honest first answer to most questions.

A great many analytical questions are fully answered here, without a model. Reach
for 04_modeling only once you can say what description could not tell you.
"""

import numpy as np
import pandas as pd


def describe_numeric(df, columns=None):
    """Count, mean, sd, variance, range and quartiles for numeric columns."""
    numeric = df[columns] if columns is not None else df.select_dtypes(include=[np.number])
    if numeric.empty:
        print('no numeric columns')
        return pd.DataFrame()
    out = numeric.describe().T
    out['variance'] = numeric.var()
    out['range'] = out['max'] - out['min']
    out['missing'] = numeric.isna().sum()
    return out[['count', 'missing', 'mean', 'std', 'variance',
                'min', '25%', '50%', '75%', 'max', 'range']].round(4)


def describe_categorical(df, columns=None, top=5):
    """Level counts and the most common value for each categorical column."""
    cats = df[columns] if columns is not None else df.select_dtypes(exclude=[np.number])
    rows = []
    for col in cats.columns:
        vc = cats[col].value_counts(dropna=True)
        if vc.empty:
            continue
        rows.append({
            'column': col,
            'levels': int(cats[col].nunique(dropna=True)),
            'missing': int(cats[col].isna().sum()),
            'most_common': vc.index[0],
            'most_common_n': int(vc.iloc[0]),
            'most_common_pct': round(100 * vc.iloc[0] / vc.sum(), 1),
            f'top_{top}': ', '.join(f'{i} ({n})' for i, n in vc.head(top).items()),
        })
    return pd.DataFrame(rows)


def group_compare(df, by, columns=None, min_n=5):
    """Mean, sd and n per group, with a warning for groups too small to compare.

    min_n only WARNS. Whether a small group is still worth reporting depends on
    what the group is; that is your call, and the warning exists so you make it
    deliberately rather than reading a mean of two rows as a finding.
    """
    numeric = (list(columns) if columns is not None
               else df.select_dtypes(include=[np.number]).columns.tolist())
    if not numeric:
        print('no numeric columns to compare')
        return pd.DataFrame()

    grouped = df.groupby(by, dropna=False)[numeric]
    out = grouped.agg(['count', 'mean', 'std']).round(4)

    sizes = df.groupby(by, dropna=False).size()
    small = sizes[sizes < min_n]
    if not small.empty:
        print(f'groups with n < {min_n}: {dict(small)}')
        print('A mean over that few rows is not a comparison. Decide whether to '
              'report, pool or exclude them, and write down why.')
    return out
