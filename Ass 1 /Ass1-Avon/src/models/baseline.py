"""Univariate regression utilities, promoted from W3/W3Act1-2/src/modeling.py.

These are TOOLS for one kind of model, not the prescribed method for any project.
Most datasets need something else, and many questions need no model at all.

compare_models() reports metrics and stops. It has no `best` field and never
picks a winner: choosing a model is a judgement about what the numbers mean for
your question, and a function that returns 'the best model' hides exactly the
decision you should be making. See the DECISION cell in 04_modeling.ipynb.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures


def build_model(X, y, degree=1):
    """Fit a LinearRegression; degree > 1 wraps it in PolynomialFeatures."""
    X = np.asarray(X, float).reshape(-1, 1)
    y = np.asarray(y, float)
    poly = None
    if degree > 1:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X = poly.fit_transform(X)
    return LinearRegression().fit(X, y), poly


def predict_with(model, poly, X):
    """Predict, applying the polynomial transform only when one was used."""
    X = np.asarray(X, float).reshape(-1, 1)
    return model.predict(poly.transform(X) if poly is not None else X)


def loocv_rmse(X, y, degree=1):
    """Leave-one-out cross-validated RMSE. Refits n times, once per held-out point.

    Out-of-sample error. Training R2 always improves with a more flexible model,
    so it cannot tell you whether the extra flexibility is real; this can.
    """
    X = np.asarray(X, float).reshape(-1, 1)
    y = np.asarray(y, float)
    n = len(y)
    errors = []
    for i in range(n):
        mask = np.ones(n, bool)
        mask[i] = False
        model, poly = build_model(X[mask], y[mask], degree)
        errors.append(y[i] - predict_with(model, poly, X[i:i + 1])[0])
    return float(np.sqrt(np.mean(np.square(errors))))


def loocv_mean_baseline(y):
    """The same LOO protocol where the prediction is the mean of the other n-1.

    Every model must beat this. A model that does not is worse than no model.
    """
    y = np.asarray(y, float)
    n = len(y)
    errors = [y[i] - np.delete(y, i).mean() for i in range(n)]
    return float(np.sqrt(np.mean(np.square(errors))))


def _fmt_coef(c):
    return f'{c:.4g}' if (abs(c) < 0.01 or abs(c) >= 1e7) else f'{c:,.4f}'


def _term(coef, name):
    sign = '-' if coef < 0 else '+'
    return f' {sign} {_fmt_coef(abs(coef))} * {name}'


def equation_text(model, degree, xname, yname):
    """The fitted equation as readable text."""
    parts = [f'{yname} = {_fmt_coef(model.intercept_)}']
    for i, coef in enumerate(model.coef_, start=1):
        parts.append(_term(coef, xname if i == 1 else f'{xname}^{i}'))
    return ''.join(parts)


def evaluate(X, y, degree, xname, yname):
    """Fit one model and return its metrics as a dict."""
    X = np.asarray(X, float).reshape(-1, 1)
    y = np.asarray(y, float)
    model, poly = build_model(X, y, degree)
    pred = predict_with(model, poly, X)
    n, k = len(y), degree
    r2 = float(r2_score(y, pred))
    adj = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else float('nan')
    return {
        'degree': degree,
        'label': 'linear' if degree == 1 else f'polynomial (deg {degree})',
        'model': model, 'poly': poly, 'n': n,
        'r2': round(r2, 4), 'adj_r2': round(float(adj), 4),
        'rmse': round(float(np.sqrt(np.mean((y - pred) ** 2))), 4),
        'loocv': round(loocv_rmse(X, y, degree), 4),
        'equation': equation_text(model, degree, xname, yname),
    }


def compare_models(data, xcol, ycol, degrees=(1, 2), min_n=4):
    """Fit each degree on complete cases and print a metrics table.

    Returns {'results': [...], 'baseline_loocv': float, 'n': int}. Deliberately
    no 'best' key. Read the LOOCV column against the baseline and decide, then
    write down why in the notebook.
    """
    sub = data[[xcol, ycol]].dropna()
    n = len(sub)
    if n < min_n:
        print(f'Only {n} complete cases for {ycol} ~ {xcol}; below min_n={min_n}. '
              'Not fitting: at this size the metrics would not mean anything.')
        return None

    X, y = sub[xcol].to_numpy(float), sub[ycol].to_numpy(float)
    results = [evaluate(X, y, d, xcol, ycol) for d in degrees]
    baseline = round(loocv_mean_baseline(y), 4)

    print(f'{ycol} ~ {xcol}   (n = {n} complete cases)')
    print('-' * 68)
    print(f'{"model":<22}{"R2 (train)":>12}{"adj R2":>10}{"LOOCV RMSE":>14}')
    print('-' * 68)
    for r in results:
        print(f'{r["label"]:<22}{r["r2"]:>12.4f}{r["adj_r2"]:>10.4f}{r["loocv"]:>14.4f}')
    print(f'{"mean-only baseline":<22}{"":>12}{"":>10}{baseline:>14.4f}')
    print('-' * 68)
    for r in results:
        print(f'  {r["label"]}: {r["equation"]}')

    print('\nReading these numbers:')
    print('  R2 (train) rises with flexibility whether or not the extra flexibility')
    print('  is real, so it cannot rank models. LOOCV RMSE is out-of-sample: lower')
    print('  is better, and anything above the baseline is worse than the mean.')
    beats = [r['label'] for r in results if r['loocv'] < baseline]
    if not beats:
        print(f'  No model here beats the mean baseline ({baseline}).')
    if n < 15:
        print(f'  n = {n} is small. Differences of this size are unlikely to be '
              'decisive; treat the ranking as weak evidence.')
    print('\nThis function does not choose. Record your choice and your reason.')

    return {'results': results, 'baseline_loocv': baseline, 'n': n,
            'xcol': xcol, 'ycol': ycol}


def plot_comparison(data, res, ax, title=''):
    """Scatter the complete cases and overlay every fitted curve in `res`.

    The mean baseline is drawn alongside the fits so the comparison that matters
    stays visible in the chart, not only in the metrics table.
    """
    from src.visualization.theme import MODEL_COLORS

    xcol, ycol = res['xcol'], res['ycol']
    sub = data[[xcol, ycol]].dropna().sort_values(xcol)
    X, y = sub[xcol].to_numpy(float), sub[ycol].to_numpy(float)
    grid = np.linspace(X.min(), X.max(), 200)

    ax.scatter(X, y, s=90, alpha=0.75, zorder=3,
               color=MODEL_COLORS['observed'], label='observed')
    styles = [('linear', '-'), ('poly', '--'), ('poly', ':')]
    for r, (role, ls) in zip(res['results'], styles):
        ax.plot(grid, predict_with(r['model'], r['poly'], grid),
                color=MODEL_COLORS[role], linewidth=2.5, linestyle=ls,
                label=f'{r["label"]} (LOOCV {r["loocv"]:,.1f})')
    ax.axhline(y.mean(), color=MODEL_COLORS['baseline'], linestyle=':', linewidth=2,
               label=f'mean baseline (LOOCV {res["baseline_loocv"]:,.1f})')

    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(title or f'{ycol} ~ {xcol}  (n = {res["n"]})')
    ax.legend(fontsize=7)
    return ax
