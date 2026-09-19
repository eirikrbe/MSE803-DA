"""Standard diagnostic and EDA charts.

Every function returns (fig, ax) so you can add annotations, then save with
save_fig(). The charts here answer generic questions -- what is missing, how is
this distributed, what correlates with what. Charts that answer YOUR question
belong in the notebook.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from src.visualization.theme import CATEGORICAL, DIVERGING, PALETTE, SEQUENTIAL, diverging, highlight


def missingness_matrix(df, figsize=(11, 5)):
    """Where the missing values are, row by row, so you can see the PATTERN.

    Scattered gaps and whole blocks mean very different things: blocks usually
    mean a subgroup was never measured, which changes whether imputing is honest.
    """
    fig, (ax_map, ax_bar) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={'width_ratios': [2.2, 1]})

    mask = df.isna().to_numpy()
    ax_map.imshow(mask, aspect='auto', interpolation='none',
                  cmap=plt.matplotlib.colors.ListedColormap(
                      [PALETTE['context'], PALETTE['alert']]))
    ax_map.set_xticks(range(df.shape[1]))
    ax_map.set_xticklabels(df.columns, rotation=45, ha='right', fontsize=7)
    ax_map.set_ylabel('row')
    ax_map.set_title('Missingness pattern (orange = missing)')
    ax_map.grid(False)

    pct = (100 * df.isna().mean()).sort_values()
    ax_bar.barh(pct.index, pct.values,
                color=[PALETTE['alert'] if v > 0 else PALETTE['context'] for v in pct],
                height=0.6)
    ax_bar.set_xlabel('% missing')
    ax_bar.set_title('Missing per column')
    ax_bar.tick_params(labelsize=7)

    fig.tight_layout()
    return fig, (ax_map, ax_bar)


def correlation_heatmap(corr, figsize=(8, 6.5), annot=True, title=None):
    """Heatmap of a correlation matrix from analysis.correlation.correlation_matrix.

    Takes the matrix rather than the raw frame, so the method you chose stays
    visible at the call site instead of being hidden in here.
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.grid(False)

    if annot:
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                v = corr.iloc[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7,
                        color='white' if abs(v) > 0.55 else PALETTE['text'])

    fig.colorbar(im, ax=ax, shrink=0.8, label='correlation')
    ax.set_title(title or 'Correlation matrix')
    fig.tight_layout()
    return fig, ax


def distribution_grid(df, columns=None, bins=20, ncols=3, figsize=None):
    """A histogram per numeric column. Look at these BEFORE any correlation.

    A correlation coefficient on a bimodal or heavily skewed variable is a number
    that describes something other than what you think it describes.
    """
    numeric = df[list(columns)] if columns is not None else df.select_dtypes(include=[np.number])
    cols = list(numeric.columns)
    if not cols:
        raise ValueError('no numeric columns to plot')

    nrows = int(np.ceil(len(cols) / ncols))
    figsize = figsize or (4.2 * ncols, 3.0 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, cols):
        s = numeric[col].dropna()
        ax.hist(s, bins=bins, color=PALETTE['accent'], edgecolor='white', linewidth=0.5)
        ax.axvline(s.mean(), color=PALETTE['alert'], linewidth=1.5,
                   label=f'mean {s.mean():.2f}')
        ax.axvline(s.median(), color=PALETTE['rule'], linewidth=1.5, linestyle='--',
                   label=f'median {s.median():.2f}')
        ax.set_title(col, fontsize=10)
        ax.legend(fontsize=7)
    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig, axes


def category_counts(df, column, top=15, figsize=(8, 5)):
    """Horizontal bar of level frequencies for one categorical column."""
    vc = df[column].value_counts(dropna=False).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh([str(i) for i in vc.index], vc.values, color=PALETTE['accent'], height=0.6)
    for i, v in enumerate(vc.values):
        ax.text(v, i, f' {v}', va='center', fontsize=8)
    ax.set_title(f'{column} - counts by level')
    ax.set_xlabel('rows')
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    return fig, ax


def ranked_bar(labels, values, subject=None, alert=None, title='', xlabel='',
               figsize=(9, 6), value_fmt='{:.2f}'):
    """Ranked horizontal bars with optional highlighting.

    `subject` and `alert` are the labels to pick out in accent and alert colour;
    everything else is muted context. Highlighting is how a chart states its point.
    """
    order = np.argsort(values)
    labels = np.asarray(labels)[order]
    values = np.asarray(values)[order]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(labels, values, color=highlight(labels, subject, alert), height=0.62)
    for i, v in enumerate(values):
        ax.text(v, i, ' ' + value_fmt.format(v), va='center', fontsize=7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    return fig, ax


def correlation_bars(correlations, title='Correlation with target', figsize=(8, 4.5)):
    """Diverging bars for a Series of correlations (accent positive, alert negative)."""
    s = correlations.sort_values()
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(s.index, s.values, color=diverging(s.values), height=0.55)
    ax.axvline(0, color=PALETTE['rule'], linewidth=1)
    ax.set_xlim(-1, 1)
    ax.set_title(title)
    ax.set_xlabel('correlation coefficient')
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    return fig, ax


def scatter_by_group(df, x, y, hue, ax=None, palette=None, figsize=(6, 5), title=None):
    """Scatter of y against x, one colour per level of a categorical `hue`.

    For 3+ co-equal categories (e.g. species) where none of PALETTE's semantic
    roles fit -- see theme.CATEGORICAL. No fit line and no computed trend, same
    as scatter_with_fit: separating classes visually is what this answers,
    nothing about whether that separation is `real` is decided in here.
    """
    sub = df[[x, y, hue]].dropna()
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    palette = palette or CATEGORICAL
    for i, level in enumerate(sorted(sub[hue].unique())):
        mask = sub[hue] == level
        ax.scatter(sub.loc[mask, x], sub.loc[mask, y], s=50, alpha=0.8,
                   color=palette[i % len(palette)], label=str(level), zorder=3)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title or f'{y} vs {x}  (n = {len(sub)})')
    ax.legend(fontsize=7, title=hue)
    if created_fig:
        fig.tight_layout()
    return fig, ax


def confusion_matrix_heatmap(cm, labels, ax=None, title=None, figsize=(6.5, 5.5)):
    """Heatmap of a confusion matrix: rows = true class, columns = predicted.

    Takes the matrix rather than raw y_true/y_pred, same reasoning as
    correlation_heatmap -- whatever you computed it with (counts, normalised)
    stays visible at the call site. Sequential colormap because counts are
    unsigned, unlike a correlation's -1..1 range.

    Pass an existing `ax` to place this inside a multi-panel figure (e.g. one
    subplot per model being compared), same convention as scatter_by_group.
    """
    cm = np.asarray(cm)
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    im = ax.imshow(cm, cmap=SEQUENTIAL)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    ax.grid(False)

    # Viridis runs dark purple (low) -> yellow (high), the opposite of the
    # diverging map correlation_heatmap uses -- so unlike there, it's the
    # HIGH values that need dark text and low values that need white.
    vmax = cm.max() if cm.size else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            ax.text(j, i, f'{v:d}', ha='center', va='center', fontsize=9,
                    color=PALETTE['text'] if v > vmax * 0.5 else 'white')

    fig.colorbar(im, ax=ax, shrink=0.8, label='count')
    ax.set_title(title or 'Confusion matrix')
    if created_fig:
        fig.tight_layout()
    return fig, ax


def svm_boundary_plot(pipeline, X2, y2, feature_names, ax=None, palette=None,
                       title=None, resolution=300):
    """Decision regions and support vectors for an SVM fit on exactly 2 features.

    The mathematical formulation calls this "a hyperplane, or set of
    hyperplanes, in a high or infinite dimensional space" -- with only 2
    input features that hyperplane is just a line or curve on paper, which
    is the only case this function can draw. `pipeline` must be fit on
    exactly `feature_names` (2 columns); a 4-feature boundary can't be
    rendered directly, which is why this is normally called on a separate,
    illustrative 2-feature refit rather than the model actually evaluated.

    `SVC` stores support_vectors_ in the SCALED space its Pipeline's scaler
    produced, so they're inverse-transformed back to the original feature
    units here -- otherwise the circles would land in standard-deviation
    units on an axis drawn in the data's own units (cm, here).
    """
    X2 = np.asarray(X2, dtype=float)
    y2 = np.asarray(y2)
    if X2.shape[1] != 2:
        raise ValueError(f'svm_boundary_plot needs exactly 2 features, got {X2.shape[1]}')

    levels = sorted(pd.unique(y2))
    palette = palette or CATEGORICAL
    color_map = {lvl: palette[i % len(palette)] for i, lvl in enumerate(levels)}

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(7, 5.5))
    else:
        fig = ax.figure

    pad_x = 0.05 * (X2[:, 0].max() - X2[:, 0].min())
    pad_y = 0.05 * (X2[:, 1].max() - X2[:, 1].min())
    xx, yy = np.meshgrid(
        np.linspace(X2[:, 0].min() - pad_x, X2[:, 0].max() + pad_x, resolution),
        np.linspace(X2[:, 1].min() - pad_y, X2[:, 1].max() + pad_y, resolution),
    )
    grid = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=list(feature_names))
    zz = np.array([levels.index(v) for v in pipeline.predict(grid)]).reshape(xx.shape)

    ax.contourf(xx, yy, zz, levels=np.arange(len(levels) + 1) - 0.5,
                cmap=ListedColormap(palette[:len(levels)]), alpha=0.22, zorder=1)

    for lvl in levels:
        mask = y2 == lvl
        ax.scatter(X2[mask, 0], X2[mask, 1], s=45, alpha=0.9, color=color_map[lvl],
                   edgecolor='white', linewidth=0.4, label=str(lvl), zorder=3)

    svm_step, scaler_step = pipeline.named_steps['svm'], pipeline.named_steps['scaler']
    support = scaler_step.inverse_transform(svm_step.support_vectors_)
    ax.scatter(support[:, 0], support[:, 1], s=150, facecolors='none',
               edgecolors=PALETTE['rule'], linewidths=1.3, zorder=4,
               label=f'support vectors (n={len(support)})')

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(title or 'SVM decision regions')
    ax.legend(fontsize=7)
    if created_fig:
        fig.tight_layout()
    return fig, ax


def scatter_with_fit(df, x, y, fit=None, figsize=(7, 5), title=None):
    """Scatter of y against x. Draws a fit line only if you pass fitted values.

    `fit` is not computed here on purpose: fitting a line is a modelling decision,
    and a trend line drawn by a plotting function is an unexamined claim about
    the relationship.
    """
    sub = df[[x, y]].dropna()
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(sub[x], sub[y], s=55, alpha=0.8, color=PALETTE['accent'], zorder=3)
    if fit is not None:
        order = np.argsort(sub[x].to_numpy())
        ax.plot(sub[x].to_numpy()[order], np.asarray(fit)[order],
                color=PALETTE['alert'], linewidth=2, zorder=4, label='fitted')
        ax.legend()
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title or f'{y} vs {x}  (n = {len(sub)})')
    fig.tight_layout()
    return fig, ax
