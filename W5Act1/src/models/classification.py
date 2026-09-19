"""Classification-side counterpart to src/models/baseline.py's regression baseline.

baseline.py's standing rule holds here too: a model is reported as a gain over
a baseline, never in isolation. `loocv_mean_baseline` is the regression version
of "what would predicting nothing but the average get you"; `majority_class_baseline`
is the classification version -- predict the training set's most frequent class
for every test row, and see how far that gets you before crediting a fitted
model with anything a naive rule already claims for free.

Deliberately does not wrap a single, already-decided SVC/cross_val_score/
confusion_matrix/classification_report call -- those are generic sklearn
calls, and the model/kernel/fold choices they carry belong visible in the
notebook, not hidden behind a helper.

compare_kernels() is the one exception, and it earns it the same way
baseline.py's compare_models() does: it is not a single decided call but a
comparison across an explicit, still-open set of choices (which kernel).
It fits every variant, prints a full metrics table, and returns no
'best'/'winner' key -- which kernel to prefer is the judgement DECISION 10
in the notebook records, not a decision this function makes.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def majority_class_baseline(y_train, y_test):
    """Predict the training set's most frequent class for every test row.

    Returns (accuracy, majority_class). The majority class is computed from
    y_train only -- using y_test to pick it would leak information a real
    predictor deployed on new rows would never have.
    """
    y_train = pd.Series(y_train).reset_index(drop=True)
    y_test = pd.Series(y_test).reset_index(drop=True)
    if y_train.empty:
        raise ValueError('y_train is empty -- no majority class to compute')
    if y_test.empty:
        raise ValueError('y_test is empty -- nothing to score against')

    majority_class = y_train.value_counts().idxmax()
    accuracy = float((y_test == majority_class).mean())
    return accuracy, majority_class


def report_baseline_gain(baseline_accuracy, majority_class, model_accuracy,
                         model_label='model', n_test=None):
    """Print model accuracy against the majority-class baseline, gain explicit.

    Mirrors the standing rule in src/models/baseline.py: no accuracy is ever
    reported in isolation, only as a gain over a baseline computed first.
    Returns the gain (model - baseline) so it can be asserted on or written
    into a report rather than re-derived from printed text.
    """
    gain = float(model_accuracy - baseline_accuracy)
    n_note = f' (n = {n_test})' if n_test is not None else ''
    print(f'majority-class baseline{n_note}: {baseline_accuracy:.4f} '
          f'(always predicting {majority_class!r})')
    print(f'{model_label} accuracy{n_note}: {model_accuracy:.4f}')
    print(f'gain over baseline: {gain:+.4f}')
    if gain <= 0:
        print(f'{model_label} does not beat the majority-class baseline -- '
              'that is a finding to report plainly, not a result to bury.')
    return gain


def cv_summary(scores):
    """Fold scores from cross_val_score, summarised with mean and std.

    Does not choose the number of folds or the splitter -- CV_FOLDS and
    StratifiedKFold stay visible as explicit choices at the call site, not
    hidden inside a wrapper.
    """
    scores = np.asarray(scores, float)
    if scores.size == 0:
        raise ValueError('scores is empty -- nothing to summarise')
    return {'scores': scores, 'mean': float(scores.mean()), 'std': float(scores.std())}


def _kernel_pipeline(kernel, poly_degree):
    """Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel=...))]).

    Scaling applies uniformly to every kernel, including linear, so the
    comparison isn't confounded by which kernel happens to also get scaled --
    RBF and polynomial kernels are scale-sensitive (they operate on distances
    / dot-products between features), but a fair comparison table needs one
    consistent pipeline shape across the row, not scaling only where it is
    known to matter.

    degree is only meaningful for kernel='poly' -- SVC ignores it otherwise --
    so it's only passed there, to avoid implying a degree was chosen for
    kernels that do not use one.
    """
    svm_kwargs = {'kernel': kernel}
    if kernel == 'poly':
        svm_kwargs['degree'] = poly_degree
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(**svm_kwargs)),
    ])


def compare_kernels(X_train, y_train, X_test, y_test, kernels=('linear', 'poly', 'rbf'),
                     cv_folds=5, random_state=None, poly_degree=3):
    """Fit an SVC per kernel inside a scaling Pipeline, print a metrics table.

    Mirrors src/models/baseline.py's compare_models(): every kernel is fit and
    scored the same way, a full table is printed, and the return has no
    'best'/'winner' key. Which kernel to prefer -- if any -- is the judgement
    DECISION 10 in the notebook records; this function reports and stops.

    Every kernel is fit inside Pipeline([('scaler', StandardScaler()),
    ('svm', SVC(kernel=...))]). Cross-validation runs the *whole pipeline* per
    fold via cross_val_score(pipeline, X_train, y_train, cv=...), so the
    scaler is refit on each fold's training portion only -- scaling on the
    full training set once, before CV, would leak each held-out fold's rows
    into the scaler's fitted mean/std.

    poly_degree defaults to sklearn's own SVC default (3), but is taken as an
    explicit parameter and passed visibly to SVC(kernel='poly', degree=...)
    rather than left as an implicit default silently inherited from sklearn.

    random_state seeds the StratifiedKFold shuffle only -- pass the notebook's
    RANDOM_STATE so the CV split is reproducible; it is not defaulted here.

    Returns {'results': [...], 'baseline_accuracy': float, 'majority_class':,
    'n_test': int}. Each entry in 'results' also carries the fitted
    'pipeline' (refit on the full X_train/y_train) and its 'y_pred' on
    X_test, so a caller building a confusion matrix or classification report
    per kernel can reuse them instead of refitting a second time.
    """
    baseline_accuracy, majority_class = majority_class_baseline(y_train, y_test)
    n_test = len(y_test)

    results = []
    for kernel in kernels:
        pipeline = _kernel_pipeline(kernel, poly_degree)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        test_accuracy = float(accuracy_score(y_test, y_pred))

        cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(_kernel_pipeline(kernel, poly_degree), X_train, y_train,
                                    cv=cv_splitter)
        cv = cv_summary(cv_scores)

        label = f'poly (degree={poly_degree})' if kernel == 'poly' else kernel
        results.append({
            'kernel': kernel,
            'label': label,
            'test_accuracy': round(test_accuracy, 4),
            'cv_mean': round(cv['mean'], 4),
            'cv_std': round(cv['std'], 4),
            'cv_scores': cv['scores'],
            'gain_over_baseline': round(test_accuracy - baseline_accuracy, 4),
            'pipeline': pipeline,
            'y_pred': y_pred,
        })

    print(f'majority-class baseline (n = {n_test}): {baseline_accuracy:.4f} '
          f'(always predicting {majority_class!r})')
    print('-' * 72)
    print(f'{"kernel":<20}{"test acc":>10}{"cv mean":>10}{"cv std":>10}{"gain":>12}')
    print('-' * 72)
    for r in results:
        print(f'{r["label"]:<20}{r["test_accuracy"]:>10.4f}{r["cv_mean"]:>10.4f}'
              f'{r["cv_std"]:>10.4f}{r["gain_over_baseline"]:>+12.4f}')
    print('-' * 72)
    print('\nThis function does not choose. Record the kernel and your reason in DECISION 10.')

    return {
        'results': results,
        'baseline_accuracy': baseline_accuracy,
        'majority_class': majority_class,
        'n_test': n_test,
    }
