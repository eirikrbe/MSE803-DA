"""Loading and saving, so paths and formats are handled in one place.

Mechanics only. Nothing here decides what the data means.
"""

import pandas as pd

from src.paths import INTERIM, PROCESSED, RAW

_READERS = {
    '.csv': pd.read_csv,
    '.tsv': lambda p, **kw: pd.read_csv(p, sep='\t', **kw),
    '.xlsx': pd.read_excel,
    '.xls': pd.read_excel,
    '.json': pd.read_json,
    '.parquet': pd.read_parquet,
}


def list_raw():
    """Every file sitting in data/raw/, so you can see what you have."""
    return sorted(p.name for p in RAW.iterdir() if p.is_file() and p.name != '.gitkeep')


def load_raw(name, **kwargs):
    """Read a file from data/raw/ by name, dispatching on its suffix.

    kwargs pass through to the underlying pandas reader, which is where dataset
    specifics belong (dtype=, parse_dates=, na_values=, sheet_name=).
    """
    path = RAW / name
    if not path.exists():
        available = list_raw()
        raise FileNotFoundError(
            f'{name!r} not found in {RAW}. '
            + (f'Available: {available}' if available else 'data/raw/ is empty.')
        )
    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(
            f'No reader for {path.suffix!r}. Known: {sorted(_READERS)}. '
            'Read it yourself with pandas and pass the frame on.'
        )
    return reader(path, **kwargs)


def save_interim(df, name):
    """Write a mid-pipeline frame to data/interim/ (regenerable, gitignored)."""
    INTERIM.mkdir(parents=True, exist_ok=True)
    path = INTERIM / name
    df.to_csv(path, index=False)
    print(f'wrote data/interim/{name}  ({len(df)} rows)')
    return path


def save_processed(df, name):
    """Write an analysis-ready frame to data/processed/ (tracked in git)."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / name
    df.to_csv(path, index=False)
    print(f'wrote data/processed/{name}  ({len(df)} rows)')
    return path
