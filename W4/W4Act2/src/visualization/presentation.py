"""The presentation layer: getting a finding to its audience.

A dashboard is not always the answer. Pick the format from who is reading and
what they need to do with it.

  Assessment, peer review, reproducibility  -> executed notebook + reports/*.md
  A written finding, a report figure        -> static PNG (theme.save_fig)
  Reader explores the values themselves     -> Plotly HTML (plotly_dashboard)
  Stakeholder using Power BI / Tableau      -> flat extract (export_for_bi)
  Interactive tool with real users          -> Streamlit; see README, not a default dep

Nothing here decides for you. build_dashboard() only assembles panels you have
already decided are worth showing.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.paths import DASHBOARDS, TABLES


def build_dashboard(panels, title='', figsize=(11, 8), height_ratios=None):
    """Compose 1-4 chart functions into one static figure (GridSpec).

    `panels` is a list of (draw_fn, subtitle); each draw_fn takes an Axes and
    draws on it. The first panel spans the full width when there are three or
    more, which is the layout W4Act1 used.

    Returns (fig, axes). Save it with theme.save_fig().
    """
    n = len(panels)
    if not 1 <= n <= 4:
        raise ValueError('build_dashboard takes 1 to 4 panels; '
                         'more than that is a report, not a dashboard')

    fig = plt.figure(figsize=figsize)
    if n == 1:
        axes = [fig.add_subplot(1, 1, 1)]
    elif n == 2:
        gs = fig.add_gridspec(1, 2, wspace=0.3)
        axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    else:
        gs = fig.add_gridspec(2, 2, height_ratios=height_ratios or [1.3, 1],
                              hspace=0.5, wspace=0.3)
        axes = [fig.add_subplot(gs[0, :])]
        axes += [fig.add_subplot(gs[1, i]) for i in range(n - 1)]

    for ax, (draw, subtitle) in zip(axes, panels):
        draw(ax)
        if subtitle:
            ax.set_title(subtitle, fontsize=11)

    if title:
        fig.suptitle(title, y=1.02)
    return fig, axes


def plotly_dashboard(traces, title='', name='dashboard', height=750,
                     subplot_titles=None, write_png=True):
    """Interactive Plotly version, written to outputs/dashboards/ as HTML (+PNG).

    `traces` is a list of (go trace, row, col). Use this when the reader needs to
    read exact values off the chart -- that is what the interactivity buys, and
    if nobody needs it, a PNG is the better deliverable.
    """
    from plotly.subplots import make_subplots

    DASHBOARDS.mkdir(parents=True, exist_ok=True)
    n_panels = len({(r, c) for _, r, c in traces})

    if n_panels <= 2:
        specs = [[{}, {}]] if n_panels == 2 else [[{}]]
        rows, cols = 1, max(n_panels, 1)
    else:
        specs = [[{'colspan': 2}, None], [{}, {}]]
        rows, cols = 2, 2

    fig = make_subplots(rows=rows, cols=cols, specs=specs,
                        row_heights=[0.55, 0.45] if rows == 2 else None,
                        subplot_titles=subplot_titles, vertical_spacing=0.15)
    for trace, r, c in traces:
        fig.add_trace(trace, row=r, col=c)

    fig.update_layout(title=title, template='plotly_white', height=height,
                      margin=dict(l=110, r=40, t=90, b=40), showlegend=False)

    html_path = DASHBOARDS / f'{name}.html'
    fig.write_html(html_path, include_plotlyjs='cdn')
    print(f'saved outputs/dashboards/{html_path.name}')

    if write_png:
        try:
            fig.write_image(DASHBOARDS / f'{name}.png', scale=2)
            print(f'saved outputs/dashboards/{name}.png')
        except Exception as exc:                      # kaleido may be unavailable
            print(f'PNG export skipped ({type(exc).__name__}); the HTML is written')
    return fig


def save_table(df, name, index=False):
    """Write a result table to outputs/tables/ as CSV, and print it as markdown.

    The markdown goes straight into reports/ -- which is why a result you intend
    to cite should come through here rather than being retyped.
    """
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / (name if name.endswith('.csv') else f'{name}.csv')
    df.to_csv(path, index=index)
    print(f'saved outputs/tables/{path.name}\n')
    try:
        print(df.to_markdown(index=index))
    except ImportError:
        print(df.to_string(index=index))
    return path


def export_for_bi(df, name, flatten_index=True):
    """Write a flat, typed extract for Power BI, Tableau or Excel.

    BI tools want one row per observation and one flat header row. An
    analysis-shaped frame -- MultiIndex columns from a groupby().agg(), a
    meaningful index, period dtypes -- imports badly or silently wrong, so this
    flattens the shape and reports the schema it produced.
    """
    out = df.copy()

    if flatten_index and isinstance(out.columns, pd.MultiIndex):
        out.columns = ['_'.join(str(p) for p in col if str(p) != '').strip('_')
                       for col in out.columns]
    if flatten_index and not isinstance(out.index, pd.RangeIndex):
        out = out.reset_index()

    for col in out.columns:
        if isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = out[col].astype(str)
        elif str(out[col].dtype).startswith('period'):
            out[col] = out[col].astype(str)

    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / (name if name.endswith('.csv') else f'{name}.csv')
    out.to_csv(path, index=False)

    print(f'saved outputs/tables/{path.name}  ({len(out)} rows x {out.shape[1]} cols)')
    print('\nschema for the BI tool:')
    for col in out.columns:
        print(f'  {col:<32} {out[col].dtype}')
    return path
