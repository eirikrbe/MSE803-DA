"""Imputation WITH a provenance record.

The idea promoted from W3: a filled value and an observed value must never be
indistinguishable downstream. Every call returns a provenance frame naming what
was filled, with what, by which method, and how much to trust it.

`method` has NO DEFAULT. Which imputation is appropriate depends on why the data
is missing -- MCAR, MAR or MNAR -- and mean-filling MNAR data produces a complete
table full of confident nonsense. Answer the missingness DECISION cell in
01_data_understanding.ipynb before calling anything here.
"""

import numpy as np
import pandas as pd

METHODS = {
    'mean': 'Fill with the column mean. Preserves the mean, shrinks the variance '
            'and weakens every correlation. Defensible only for a few MCAR values.',
    'median': 'Fill with the column median. As above but robust to skew and outliers.',
    'mode': 'Fill with the most frequent value. For categoricals; inflates the '
            'majority class.',
    'constant': 'Fill with a value you supply (pass fill_value=). Use when missing '
                'means something specific, e.g. 0 sales rather than unknown sales.',
    'linear_regression': 'Predict from other columns with a linear fit. Needs a real '
                         'relationship and complete predictors on the target rows; '
                         'understates uncertainty because it imputes the conditional mean.',
}


def _require_method(method):
    if method is None:
        raise ValueError(
            'method is required -- imputation is an analytical decision, not a default.\n'
            'Choose one and record why:\n' +
            '\n'.join(f'  {k}: {v}' for k, v in METHODS.items()))
    if method not in METHODS:
        raise ValueError(f'method={method!r} unknown. Available:\n' +
                         '\n'.join(f'  {k}: {v}' for k, v in METHODS.items()))


def impute(df, column, method=None, fill_value=None, predictors=None,
           id_col=None, confidence=None):
    """Fill missing values in one column, returning (filled_df, provenance).

    method       one of METHODS; required
    fill_value   for method='constant'
    predictors   for method='linear_regression'; the columns to predict from
    id_col       a column naming each row, so provenance is readable
    confidence   'high' / 'medium' / 'low'; your assessment, not a computed one

    The provenance frame has one row per filled cell and is meant to be written
    to outputs/tables/ next to the imputed data.
    """
    _require_method(method)
    if column not in df.columns:
        raise KeyError(f'{column!r} is not a column')

    out = df.copy()
    missing_mask = out[column].isna()
    n_missing = int(missing_mask.sum())
    if n_missing == 0:
        print(f'{column!r} has no missing values; nothing to impute')
        return out, pd.DataFrame()

    if method in ('mean', 'median'):
        value = out[column].mean() if method == 'mean' else out[column].median()
        out.loc[missing_mask, column] = value
        filled = pd.Series(value, index=out.index[missing_mask])

    elif method == 'mode':
        modes = out[column].mode(dropna=True)
        if modes.empty:
            raise ValueError(f'{column!r} has no non-null values to take a mode from')
        value = modes.iloc[0]
        out.loc[missing_mask, column] = value
        filled = pd.Series(value, index=out.index[missing_mask])

    elif method == 'constant':
        if fill_value is None:
            raise ValueError("method='constant' needs fill_value=")
        out.loc[missing_mask, column] = fill_value
        filled = pd.Series(fill_value, index=out.index[missing_mask])

    elif method == 'linear_regression':
        if not predictors:
            raise ValueError("method='linear_regression' needs predictors=[...]")
        from sklearn.linear_model import LinearRegression

        train = df[df[column].notna()].dropna(subset=predictors)
        target_rows = df[missing_mask].dropna(subset=predictors)
        if len(train) < len(predictors) + 2:
            raise ValueError(
                f'only {len(train)} complete training rows for {len(predictors)} '
                'predictor(s). Too few to fit a model you could defend.')
        if target_rows.empty:
            raise ValueError(
                f'no rows missing {column!r} have all of {predictors} present. '
                'Regression imputation needs the predictors on the same row.')

        model = LinearRegression().fit(train[predictors], train[column])
        preds = model.predict(target_rows[predictors])
        out.loc[target_rows.index, column] = preds
        filled = pd.Series(preds, index=target_rows.index)

        r2 = model.score(train[predictors], train[column])
        print(f'training R2 = {r2:.4f} on {len(train)} rows. '
              'Training fit is not validation -- check it out of sample before '
              'trusting these values.')
        if len(target_rows) < n_missing:
            print(f'{n_missing - len(target_rows)} of {n_missing} missing value(s) '
                  'were left as NaN: their predictors are missing too.')

    flag_col = f'{column}_was_imputed'
    out[flag_col] = out.index.isin(filled.index)

    provenance = pd.DataFrame({
        'row': filled.index,
        'id': (df.loc[filled.index, id_col].to_numpy() if id_col else filled.index),
        'column': column,
        'imputed_value': filled.to_numpy(),
        'method': method,
        'predictors': ', '.join(predictors) if predictors else '',
        'confidence': confidence or 'unrecorded',
    })

    print(f'imputed {len(filled)} value(s) in {column!r} using {method}')
    print(f'added flag column {flag_col!r}')
    if confidence is None:
        print('confidence is unrecorded. Set confidence= so a reader knows how '
              'much weight these values carry.')
    return out, provenance


def impute_with_model(df, column, predictors, method=None, id_col=None,
                      confidence=None):
    """Regression imputation. Thin wrapper over impute() for the model-based path."""
    _require_method(method)
    return impute(df, column, method=method, predictors=predictors,
                  id_col=id_col, confidence=confidence)


def write_provenance_artifacts(df_clean, df_imputed, provenance, stem):
    """Write the three artifacts separately, because they have different provenance.

    The W3 pattern. A cleaned table (observed values only), the model output (what
    was predicted and how confident), and the combined table are three different
    things, and merging them into one file loses the distinction permanently.
    """
    from src.paths import PROCESSED, TABLES

    PROCESSED.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    paths = {
        'cleaned': PROCESSED / f'{stem}_cleaned.csv',
        'predictions': TABLES / f'{stem}_imputation_provenance.csv',
        'imputed': PROCESSED / f'{stem}_cleaned_imputed.csv',
    }
    df_clean.to_csv(paths['cleaned'], index=False)
    provenance.to_csv(paths['predictions'], index=False)
    df_imputed.to_csv(paths['imputed'], index=False)

    for label, p in paths.items():
        print(f'{label:<12} -> {p.relative_to(PROCESSED.parents[1])}')
    return paths
