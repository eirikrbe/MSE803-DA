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


def cv_table(models, X, y, cv, scoring):
    """Cross-validate every model in `models` (name -> estimator) and report the
    mean and standard deviation of each score across folds.

    `cv` and `scoring` are required: the splitter and the metrics are choices
    the notebook states, not defaults this function supplies. Scores that
    sklearn reports as negative errors ('neg_root_mean_squared_error') are
    flipped back to positive so the table reads naturally.

    Returns one row per model and no winner.
    """
    rows = []
    for name, model in models.items():
        res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        row = {'model': name}
        for key in scoring:
            vals = res[f'test_{key}']
            if str(scoring[key]).startswith('neg_'):
                vals = -vals
            row[f'{key}_mean'] = float(np.mean(vals))
            row[f'{key}_sd'] = float(np.std(vals, ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


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


def interval_coverage(y_true, lower, upper):
    """Share of observations inside their prediction interval, and the mean
    interval width. A nominal 80% interval should cover about 80% of new data;
    much less means the intervals are overconfident."""
    y_true = np.asarray(y_true, float)
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    inside = (y_true >= lower) & (y_true <= upper)
    return {'coverage': float(inside.mean()), 'mean_width': float(np.mean(upper - lower))}
