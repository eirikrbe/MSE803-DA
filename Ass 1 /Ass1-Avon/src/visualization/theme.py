"""One place for colour and style, so every figure in a project matches.

The palette is semantic: colours are named for the ROLE they play, not the hue.
Promoted from W4/W4Act1, which established the convention in code comments.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

from src.paths import FIGURES

PALETTE = {
    'accent':   '#2a78d6',   # the subject of the chart / positive values
    'alert':    '#eb6834',   # the outlier / negative values
    'context':  '#b7bbad',   # everything else, the background cohort
    'baseline': '#9a9990',   # the comparison baseline
    'rule':     '#5b5c50',   # zero lines and axis rules
    'grid':     '#e0e0e0',   # gridlines
    'text':     '#2b2b2b',
}

# Model roles, promoted from W3's plot_fits so a fitted-model chart reads the same
# way across projects.
MODEL_COLORS = {
    'observed': '#1f77b4',
    'linear':   '#2ca02c',
    'poly':     '#d62728',
    'baseline': '#9a9990',
}

SEQUENTIAL = 'viridis'
DIVERGING = 'RdBu_r'

FIG_DPI = 150


def set_plot_style():
    """Apply the house style globally. Call once in the first cell of a notebook.

    Hoists what W3 and W4 were repeating on every axes by hand: no top/right
    spines, left-aligned bold titles, small ticks, a light grid behind the data.
    """
    mpl.rcParams.update({
        'figure.dpi': 100,
        'savefig.dpi': FIG_DPI,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.titlelocation': 'left',
        'axes.labelsize': 10,
        'axes.edgecolor': PALETTE['rule'],
        'axes.labelcolor': PALETTE['text'],
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.color': PALETTE['grid'],
        'grid.linewidth': 0.8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'xtick.color': PALETTE['text'],
        'ytick.color': PALETTE['text'],
        'legend.fontsize': 8,
        'legend.frameon': False,
        'figure.titlesize': 15,
        'figure.titleweight': 'bold',
        'text.color': PALETTE['text'],
    })


def save_fig(fig, name, dpi=FIG_DPI):
    """Save to outputs/figures/<name>.png at the project standard (150 dpi, tight).

    Standardised because the repo currently has three different DPIs across three
    weeks. Returns the path so you can reference it in a report.
    """
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / (name if name.endswith('.png') else f'{name}.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f'saved outputs/figures/{path.name}')
    return path


def highlight(values, subject=None, alert=None):
    """Build a per-bar colour list: subject in accent, alert in alert, rest muted.

    The pattern W4Act1 used as a nested ternary, made reusable. `subject` and
    `alert` are collections of values to pick out.
    """
    # `if x is not None` rather than `x or ()`: a pandas Series has no truth value,
    # and passing one (df['country'].head(3)) is the natural way to call this.
    subject = set(subject) if subject is not None else set()
    alert = set(alert) if alert is not None else set()
    return [PALETTE['alert'] if v in alert
            else PALETTE['accent'] if v in subject
            else PALETTE['context'] for v in values]


def diverging(values):
    """Accent for values >= 0, alert for negatives. For correlation bars."""
    return [PALETTE['accent'] if v >= 0 else PALETTE['alert'] for v in values]
