"""Data-quality diagnostics.

Every function here REPORTS. None of them mutates a frame or drops a row.
Flagging a suspicious value is mechanical and safe to automate; deciding it is an
error rather than a real observation is judgement, and belongs to you. See the
DECISION cells in notebooks/01_data_understanding.ipynb.
"""

import numpy as np
import pandas as pd


def profile(df):
    """Shape, memory and a dtype tally. The first thing to run on anything."""
    print(f'rows: {len(df):,}   columns: {df.shape[1]}')
    print(f'memory: {df.memory_usage(deep=True).sum() / 1024:,.1f} KB')
    print('\ndtypes:')
    print(df.dtypes.value_counts().to_string())
    return df.head()


def schema_report(df):
    """One row per column: dtype, non-null count, nulls, unique values, a sample.

    Read this before anything else. A column typed 'object' that you expected to
    be numeric is the single most common source of silent breakage downstream.
    """
    rows = []
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        rows.append({
            'column': col,
            'dtype': str(s.dtype),
            'non_null': int(s.notna().sum()),
            'missing': int(s.isna().sum()),
            'missing_pct': round(100 * s.isna().mean(), 2),
            'unique': int(s.nunique(dropna=True)),
            'example': non_null.iloc[0] if len(non_null) else pd.NA,
        })
    return pd.DataFrame(rows)


def missing_report(df):
    """Missing count and share per column, worst first, plus the row-level picture.

    Counting missingness is mechanical. Explaining WHY it is missing (MCAR, MAR,
    MNAR) is a judgement that determines whether imputation is legitimate at all.
    """
    out = pd.DataFrame({
        'missing': df.isna().sum(),
        'missing_pct': (100 * df.isna().mean()).round(2),
    }).sort_values('missing', ascending=False)
    out = out[out['missing'] > 0]

    complete_rows = int(df.notna().all(axis=1).sum())
    print(f'complete rows: {complete_rows:,} of {len(df):,} '
          f'({100 * complete_rows / len(df):.1f}%)')
    if out.empty:
        print('no missing values')
    return out


def duplicate_report(df, subset=None):
    """Exact duplicate rows, and near-duplicates on a key if you name one.

    `subset` is deliberately not guessed: which columns identify a record is a
    fact about your data, not something to infer from the frame.
    """
    exact = int(df.duplicated().sum())
    print(f'exact duplicate rows: {exact}')
    result = {'exact_duplicates': exact}
    if subset is not None:
        on_key = int(df.duplicated(subset=subset).sum())
        print(f'duplicate on {subset}: {on_key}')
        result['duplicates_on_subset'] = on_key
    return result


def outlier_report(df, iqr_multiplier=1.5, z_threshold=3.0):
    """Flag numeric values outside the IQR fence and beyond a z-score threshold.

    REPORTS ONLY. Nothing is removed. The two methods disagree by design: IQR is
    distribution-free, z-scores assume roughly normal data, and a value both flag
    is worth a look. A flagged value may be a data-entry error or the most
    interesting observation in the set -- that call is yours to make and record.
    """
    numeric = df.select_dtypes(include=[np.number])
    rows = []
    for col in numeric.columns:
        s = numeric[col].dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
        n_iqr = int(((s < low) | (s > high)).sum())
        sd = s.std()
        n_z = int((((s - s.mean()).abs() / sd) > z_threshold).sum()) if sd > 0 else 0
        rows.append({
            'column': col, 'n': len(s),
            'iqr_flagged': n_iqr, 'z_flagged': n_z,
            'lower_fence': round(low, 4), 'upper_fence': round(high, 4),
            'min': s.min(), 'max': s.max(),
        })
    out = pd.DataFrame(rows)
    if not out.empty and out[['iqr_flagged', 'z_flagged']].to_numpy().sum() > 0:
        print('Values flagged. Flagged is not the same as wrong -- decide and record why.')
    return out


def cardinality_report(df, high_card_ratio=0.9):
    """Distinct-value counts for non-numeric columns.

    Surfaces the shape of your categoricals: a column with one value carries no
    information, and one with a distinct value per row is an identifier, not a
    feature. Both are reported, neither is acted on.
    """
    numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
    rows = []
    for col in df.columns:
        n_unique = int(df[col].nunique(dropna=True))
        ratio = n_unique / len(df) if len(df) else 0
        note = ''
        if n_unique <= 1:
            note = 'constant - no information'
        elif ratio >= high_card_ratio and col not in numeric_cols:
            # Only flagged for non-numeric columns: a continuous measurement is
            # near-unique by nature, and calling it an identifier is noise.
            note = 'near-unique - likely an identifier'
        rows.append({'column': col, 'unique': n_unique,
                     'unique_ratio': round(ratio, 3), 'note': note})
    return pd.DataFrame(rows).sort_values('unique', ascending=False)


def quality_report(df, subset=None):
    """Run every diagnostic above and return the per-column summary as one frame.

    This is the mechanical pass: safe to run on any tabular dataset, on day one,
    before you know anything about it. What to DO about what it finds is the part
    it will not decide for you.
    """
    print('=' * 66)
    print('PROFILE')
    print('=' * 66)
    profile(df)

    print('\n' + '=' * 66)
    print('MISSING')
    print('=' * 66)
    missing = missing_report(df)
    if not missing.empty:
        print(missing.to_string())

    print('\n' + '=' * 66)
    print('DUPLICATES')
    print('=' * 66)
    duplicate_report(df, subset=subset)

    print('\n' + '=' * 66)
    print('OUTLIERS (flagged, not removed)')
    print('=' * 66)
    outliers = outlier_report(df)
    print(outliers.to_string(index=False) if not outliers.empty
          else 'no numeric columns with enough data')

    schema = schema_report(df)
    card = cardinality_report(df)[['column', 'unique_ratio', 'note']]
    summary = schema.merge(card, on='column', how='left')
    if not outliers.empty:
        summary = summary.merge(
            outliers[['column', 'iqr_flagged', 'z_flagged']], on='column', how='left')

    print('\n' + '=' * 66)
    print('SUMMARY (one row per column)')
    print('=' * 66)
    print(summary.to_string(index=False))
    return summary
