"""Model evaluation: baseline first, out of sample always, uncertainty stated.

The standing rules of this template hold here:

- A model is reported as a gain over a baseline, never in isolation.
- Training-set scores are not evidence. Every number comes from cross-validation
  folds or from a held-out test set.
- A point estimate travels with its uncertainty: fold standard deviations for
  cross-validation, bootstrap intervals for the test set.

Nothing here picks a winner. `cv_table` reports every candidate side by side and
returns no 'best' key, because which model to prefer is a judgement that weighs
accuracy against interpretability, cost and risk (DECISION 10 in the notebooks).
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate


def regression_metrics(y_true, y_pred):
    """RMSE, MAE and R² for one set of predictions, as a dict."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true, y_pred, proba=None, labels=None):
    """Accuracy, balanced accuracy, macro precision / recall / F1 and, given
    class probabilities, the one-vs-rest macro ROC-AUC.

    Macro averages weight every class equally, so a model cannot look good by
    serving only the largest class. `proba` columns must follow `labels`.
    """
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0)
    out = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'precision_macro': float(p),
        'recall_macro': float(r),
        'f1_macro': float(f),
    }
    if proba is not None:
        proba = np.asarray(proba, float)
        if labels is not None:
            # roc_auc_score wants labels in sorted order; reorder the columns to match,
            # so callers can keep a natural order such as low -> high.
            order = np.argsort(np.asarray(labels, dtype=object).astype(str))
            labels = [labels[i] for i in order]
            proba = proba[:, order]
        out['roc_auc_ovr'] = float(roc_auc_score(
            y_true, proba, multi_class='ovr', average='macro', labels=labels))
    return out


def per_class_report(y_true, y_pred, labels):
    """Precision, recall, F1 and support for each class, in `labels` order."""
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame({'class': labels, 'precision': p, 'recall': r,
                         'f1': f, 'support': s})


def normalised_confusion(y_true, y_pred, labels):
    """Confusion matrix with each row summing to 1: the share of each true class
    that landed in each predicted class. Rows are recall, the diagonal is
    per-class recall."""
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
    cm = cm / cm.sum(axis=1, keepdims=True)
    return pd.DataFrame(cm, index=pd.Index(labels, name='true'),
                        columns=pd.Index(labels, name='predicted'))


def bootstrap_ci(y_true, y_pred, metric, B=2000, seed=0, level=0.95):
    """Percentile bootstrap interval for any metric(y_true, y_pred) on a test set.

    Resamples test rows with replacement, so the interval reflects how much the
    score would move on a different test sample of the same size. It does not
    capture training variability; cross-validation fold spread does that.
    Returns (point, low, high).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        raise ValueError('empty test set')
    rng = np.random.default_rng(seed)
    point = float(metric(y_true, y_pred))
    draws = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        draws[b] = metric(y_true[idx], y_pred[idx])
    tail = (1 - level) / 2
    return point, float(np.quantile(draws, tail)), float(np.quantile(draws, 1 - tail))


def cv_table(models, X, y, cv, scoring, return_train=False):
    """Cross-validate every model in `models` (name -> estimator) and report the
    mean and standard deviation of each score across folds.

    `cv` and `scoring` are required: the splitter and the metrics are choices
    the notebook states, not defaults this function supplies. Scores that
    sklearn reports as negative errors ('neg_root_mean_squared_error') are
    flipped back to positive so the table reads naturally.

    With `return_train=True`, each score also gets a `<key>_train_mean` column:
    the mean score on the training folds. It is a diagnostic only. A large gap
    between it and the validation mean shows memorisation; it is never evidence
    of performance.

    Returns one row per model and no winner.
    """
    rows = []
    for name, model in models.items():
        res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1,
                             return_train_score=return_train)
        row = {'model': name}
        for key in scoring:
            sign = -1 if str(scoring[key]).startswith('neg_') else 1
            vals = sign * res[f'test_{key}']
            row[f'{key}_mean'] = float(np.mean(vals))
            row[f'{key}_sd'] = float(np.std(vals, ddof=1))
            if return_train:
                row[f'{key}_train_mean'] = float(np.mean(sign * res[f'train_{key}']))
        rows.append(row)
    return pd.DataFrame(rows)


def one_se_choice(cv, metric, simplicity_order, n_folds, higher_is_better=True):
    """Apply the one-standard-error rule to a `cv_table` result.

    Every model whose mean validation score is within one standard error of the
    best is treated as tied with it. From the tied models, the first one in
    `simplicity_order` (simplest first: the order is stated before the scores
    are seen) is chosen (James et al., 2021, section 6.1.3). Only models named in
    `simplicity_order` compete, so baselines and reference ceilings can stay in
    the table without being chosen.

    The standard error is the best model's fold SD / sqrt(n_folds). The rule
    uses training folds only, so the test set plays no part in the choice.

    Returns (chosen name, the table of candidates with a `tied_with_best` column).
    """
    cand = cv[cv['model'].isin(simplicity_order)].copy()
    if cand.empty:
        raise ValueError('no model in simplicity_order appears in the table')
    mean, sd = cand[f'{metric}_mean'], cand[f'{metric}_sd']
    best = mean.idxmax() if higher_is_better else mean.idxmin()
    se = sd[best] / np.sqrt(n_folds)
    bound = mean[best] - se if higher_is_better else mean[best] + se
    cand['tied_with_best'] = mean >= bound if higher_is_better else mean <= bound
    cand['simplicity_rank'] = cand['model'].map({m: i for i, m in enumerate(simplicity_order)})
    cand = cand.sort_values('simplicity_rank')
    chosen = cand.loc[cand['tied_with_best'], 'model'].iat[0]
    return chosen, cand


def mcnemar(y_true, pred_a, pred_b):
    """Exact McNemar test: do two classifiers make different errors on the same
    test rows? Uses only the discordant pairs, so it is the right paired test
    when both models were scored on one test set.

    Returns b (A right, B wrong), c (A wrong, B right) and the two-sided exact
    binomial p-value.
    """
    y_true = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == y_true
    b_ok = np.asarray(pred_b) == y_true
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    p = 1.0 if b + c == 0 else float(stats.binomtest(b, b + c, 0.5).pvalue)
    return {'a_right_b_wrong': b, 'a_wrong_b_right': c, 'p_value': p}


def wilson_ci(successes, n, level=0.95):
    """Wilson score interval for a proportion. It stays inside [0, 1] and holds
    up better than the normal approximation near 0 or 1 and for small n."""
    if n == 0:
        raise ValueError('n must be positive')
    z = stats.norm.ppf(1 - (1 - level) / 2)
    p = successes / n
    centre = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return float(centre - half), float(centre + half)


def interval_coverage(y_true, lower, upper, level=0.95):
    """Share of observations inside their prediction interval, with a Wilson
    interval for that share, and the mean interval width. A nominal 80%
    interval should cover about 80% of new data; much less means the intervals
    are overconfident."""
    y_true = np.asarray(y_true, float)
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    inside = (y_true >= lower) & (y_true <= upper)
    lo, hi = wilson_ci(int(inside.sum()), len(inside), level)
    return {'coverage': float(inside.mean()), 'coverage_low': lo, 'coverage_high': hi,
            'mean_width': float(np.mean(upper - lower))}


def calibration_summary(y_true, p, n_bins=10):
    """How far predicted probabilities can be taken at face value, for one
    binary outcome.

    Returns the Brier score, the calibration intercept and slope (a logistic
    regression of the outcome on the logit of `p`; 0 and 1 mean perfectly
    calibrated, a slope below 1 means overconfident), and a reliability table
    of mean predicted against observed rate in `n_bins` equal-count bins.
    """
    import statsmodels.api as sm
    y = np.asarray(y_true, float)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    fit = sm.GLM(y, sm.add_constant(logit), family=sm.families.Binomial()).fit()
    bins = pd.qcut(p, n_bins, labels=False, duplicates='drop')
    table = (pd.DataFrame({'bin': bins, 'p': p, 'y': y}).groupby('bin')
             .agg(n=('y', 'size'), mean_predicted=('p', 'mean'), observed=('y', 'mean')).reset_index())
    return {'brier': float(np.mean((p - y) ** 2)),
            'intercept': float(fit.params[0]), 'slope': float(fit.params[1])}, table


def cost_threshold(y_true, p, miss_cost, grid):
    """The threshold on `p` that minimises miss_cost x missed + false alarms.

    Ties go to the higher threshold, so equal-cost options flag fewer people.
    Choose it on validation or out-of-fold predictions, never on the test set
    that will then report its performance.
    """
    y = np.asarray(y_true).astype(bool)
    p = np.asarray(p, float)
    best_cost, best_t = None, None
    for t in sorted(grid, reverse=True):
        flag = p >= t
        cost = miss_cost * int((~flag & y).sum()) + int((flag & ~y).sum())
        if best_cost is None or cost < best_cost:
            best_cost, best_t = cost, float(t)
    return best_t


def threshold_report(y_true, p, t):
    """Flags, precision, recall, false alarms and misses at threshold `t`."""
    y = np.asarray(y_true).astype(bool)
    flag = np.asarray(p, float) >= t
    tp, fp, fn = int((flag & y).sum()), int((flag & ~y).sum()), int((~flag & y).sum())
    return {'threshold': float(t), 'flagged': int(flag.sum()), 'precision': tp / max(tp + fp, 1),
            'recall': tp / max(tp + fn, 1), 'false_alarms': fp, 'missed': fn}
