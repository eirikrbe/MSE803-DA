"""Modelling, evaluation and validation helpers for the W4Act1 analysis.

Extracted from W4Act1_statistics.ipynb once the logic stabilised. Every function
here is univariate by design (one predictor -> one target), matching the structure
of the course sample code in scripts/.

Evaluation deliberately reports adjusted R^2, leave-one-out CV and a mean-only
baseline alongside R^2, because training R^2 cannot fall when a polynomial term is
added and therefore cannot be used to choose between nested models.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error


def build_model(X, y, degree):
    # Fit a univariate model. degree=1 -> plain linear, degree=2 -> teacher's PolynomialFeatures.
    poly = PolynomialFeatures(degree=degree, include_bias=False) if degree > 1 else None
    X_design = poly.fit_transform(X) if poly is not None else X
    model = LinearRegression().fit(X_design, y)
    return model, poly


def predict_with(model, poly, X):
    return model.predict(poly.transform(X) if poly is not None else X)


def loocv_rmse(X, y, degree):
    # Leave-one-out CV: refit n times, each time predicting the point that was held out.
    errors = []
    for i in range(len(X)):
        keep = np.ones(len(X), dtype=bool)
        keep[i] = False
        model, poly = build_model(X[keep], y[keep], degree)
        errors.append((y[i] - predict_with(model, poly, X[i:i + 1])[0]) ** 2)
    return float(np.sqrt(np.mean(errors)))


def loocv_mean_baseline(y):
    # Same protocol, but the "model" is just the mean of the other n-1 points.
    errors = [(y[i] - np.delete(y, i).mean()) ** 2 for i in range(len(y))]
    return float(np.sqrt(np.mean(errors)))


def _fmt_coef(c):
    # Keep tiny coefficients readable - 'Salary^2' coefficients are ~1e-8, not '0.0000'
    a = abs(c)
    if a == 0:
        return "0"
    if a < 0.01 or a >= 1e7:
        return f"{a:.4g}"
    return f"{a:,.4f}"


def _term(coef, name):
    return f" {'-' if coef < 0 else '+'} {_fmt_coef(coef)} * {name}"


def equation_text(model, degree, xname, yname):
    eq = f"{yname} = {model.intercept_:,.2f}" + _term(model.coef_[0], xname)
    if degree > 1:
        eq += _term(model.coef_[1], f"{xname}^2")
    return eq


def evaluate(X, y, degree, xname, yname):
    model, poly = build_model(X, y, degree)
    y_pred = predict_with(model, poly, X)
    n, p = len(y), degree
    r2 = r2_score(y, y_pred)
    return {
        'degree': degree,
        'label': 'Linear' if degree == 1 else 'Polynomial (deg 2)',
        'model': model, 'poly': poly, 'n': n,
        'r2': r2,
        'adj_r2': 1 - (1 - r2) * (n - 1) / (n - p - 1) if n - p - 1 > 0 else np.nan,
        'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
        'loocv': loocv_rmse(X, y, degree),
        'equation': equation_text(model, degree, xname, yname),
    }


def compare_models(data, xcol, ycol, title):
    # Fit linear + polynomial on the complete cases and print an honest side-by-side comparison.
    sub = data[[xcol, ycol]].dropna()
    X = sub[[xcol]].values
    y = sub[ycol].values

    print("=" * 72)
    print(f"{title}   ({ycol} <- {xcol})")
    print("=" * 72)
    print(f"Complete cases available for training: n = {len(sub)}")
    if len(sub) < 4:
        print("Too few complete cases to fit anything meaningful.")
        return None

    results = [evaluate(X, y, 1, xcol, ycol), evaluate(X, y, 2, xcol, ycol)]
    baseline = loocv_mean_baseline(y)

    print(f"\n{'Model':<20}{'R^2':>10}{'Adj R^2':>11}{'Train RMSE':>14}{'LOOCV RMSE':>14}")
    print("-" * 72)
    for r in results:
        print(f"{r['label']:<20}{r['r2']:>10.4f}{r['adj_r2']:>11.4f}"
              f"{r['rmse']:>14,.0f}{r['loocv']:>14,.0f}")
    print(f"{'Mean-only baseline':<20}{0.0:>10.4f}{'-':>11}{y.std(ddof=0):>14,.0f}{baseline:>14,.0f}")

    print("\nFitted equations:")
    for r in results:
        print(f"  {r['label']:<20} {r['equation']}")

    # Honest verdict - never decided on training R^2 alone
    lin, poly = results
    print("\nREADING THE NUMBERS:")
    direction = 'rose' if poly['r2'] > lin['r2'] else 'FELL'
    print(f"  - Training R^2 {direction} {lin['r2']:.4f} -> {poly['r2']:.4f} when {xcol}^2 was added.")
    if poly['r2'] >= lin['r2']:
        print(f"    A rise is guaranteed by OLS (the linear model is the polynomial with b2 = 0,")
        print(f"    so least squares can always match it and will edge past it by fitting noise).")
        print(f"    It is therefore arithmetic, NOT evidence of a curved relationship.")
    else:
        print(f"    In exact arithmetic this is IMPOSSIBLE: the linear model is the polynomial with")
        print(f"    b2 = 0, so the polynomial can never fit worse. Seeing R^2 fall means the")
        print(f"    least-squares solve is numerically ill-conditioned - {xcol} runs to ~{X.max():,.0f}")
        print(f"    while {xcol}^2 runs to ~{X.max() ** 2:,.0f}, five orders of magnitude apart, and the")
        print(f"    solver degrades. The fitted polynomial has collapsed toward a flat line.")
        print(f"    This is a further reason to reject the polynomial for this predictor.")
    print(f"  - Adjusted R^2 went {lin['adj_r2']:.4f} -> {poly['adj_r2']:.4f} "
          f"({'the extra term did NOT pay for itself' if poly['adj_r2'] < lin['adj_r2'] else 'the extra term paid for itself'}).")
    print(f"  - LOOCV RMSE went {lin['loocv']:,.0f} -> {poly['loocv']:,.0f} "
          f"({'polynomial predicts unseen points WORSE' if poly['loocv'] > lin['loocv'] else 'polynomial predicts unseen points better'}).")
    better = lin if lin['loocv'] <= poly['loocv'] else poly
    print(f"  => Preferred on out-of-sample error: {better['label']}")
    if better['loocv'] < baseline:
        print(f"  => It beats the mean-only baseline ({better['loocv']:,.0f} < {baseline:,.0f}), "
              f"by {100 * (1 - better['loocv'] / baseline):.1f}% - a real but small signal.")
    else:
        print(f"  => WARNING: it does NOT beat the mean-only baseline ({better['loocv']:,.0f} >= {baseline:,.0f}).")
        print(f"     Predicting the average would be more accurate than using this predictor.")
    print(f"  => At n = {len(sub)}, none of these differences are statistically decisive.")

    return {'data': sub, 'X': X, 'y': y, 'linear': lin, 'poly': poly,
            'baseline': baseline, 'best': better, 'xcol': xcol, 'ycol': ycol}


def plot_fits(res, ax, title):
    X, y = res['X'], res['y']
    ax.scatter(X, y, color='#1f77b4', s=90, alpha=0.75, zorder=3, label='Observed data')
    grid = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    ax.plot(grid, predict_with(res['linear']['model'], res['linear']['poly'], grid),
            color='#2ca02c', lw=2.5, label=f"Linear (LOOCV {res['linear']['loocv']:,.0f})")
    ax.plot(grid, predict_with(res['poly']['model'], res['poly']['poly'], grid),
            color='#d62728', lw=2.5, ls='--', label=f"Poly deg 2 (LOOCV {res['poly']['loocv']:,.0f})")
    ax.axhline(y.mean(), color='grey', lw=1.5, ls=':',
               label=f"Mean-only (LOOCV {res['baseline']:,.0f})")
    ax.set_xlabel(res['xcol'], fontsize=11)
    ax.set_ylabel(res['ycol'], fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def permutation_test(data, xcol, ycol, B=2000, seed=0):
    """Shuffle the target B times and count how often noise beats the real model.

    Returns (real_gain, null_gains, p_value) where gain = mean-only baseline RMSE
    minus model LOOCV RMSE, so larger is better.

    Preferred over bootstrap resampling here: resampling with replacement
    duplicates rows, and a duplicated row can land on both sides of a leave-one-out
    split, leaking the answer and flattering the model. Permutation never
    duplicates a row.
    """
    sub = data[[xcol, ycol]].dropna()
    X, y = sub[[xcol]].values, sub[ycol].values
    real_gain = loocv_mean_baseline(y) - loocv_rmse(X, y, 1)
    rng = np.random.default_rng(seed)
    null_gains = np.array([loocv_mean_baseline(yp) - loocv_rmse(X, yp, 1)
                           for yp in (rng.permutation(y) for _ in range(B))])
    return real_gain, null_gains, float((null_gains >= real_gain).mean())
