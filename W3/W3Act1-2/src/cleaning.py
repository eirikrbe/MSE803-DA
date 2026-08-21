"""Data-cleaning helpers for the W3Act1-2 dataset.

Extracted from W3Act1-2.ipynb once the logic stabilised, so it can be
imported by the notebook and covered by tests in tests/test_cleaning.py.
"""

import numpy as np
import pandas as pd

# Spelled-out number vocabulary. Deliberately small - this dataset only contains
# tens and units, and a wider vocabulary would need a real parser, not a lookup.
word_to_num = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
    'ten': 10, 'twenty': 20, 'thirty': 30, 'forty': 40,
    'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
}


def text_to_number(text):
    """'sixty five' -> 65, 'thirty-eight' -> 38, 'sixty-five thousand' -> 65000.

    Hyphens are treated as spaces, so both written forms parse identically.
    Returns None when no number word is recognised.
    """
    tokens = str(text).lower().replace('-', ' ').split()
    total = sum(word_to_num[t] for t in tokens if t in word_to_num)
    if 'thousand' in tokens:
        total *= 1000
    return total if total > 0 else None


def clean_number(val):
    """Parse a numeric cell however it was written: 38, '30,000', or 'thirty-eight'.

    ONE parser for Age, Salary and Net worth. An earlier version used a per-column
    lookup for Age ({'thirty-eight': 38}), which silently returned NaN for every
    other spelled-out age - a bug this dataset hid, because it happens to contain
    exactly one such value per column, both in forms that happened to work.
    """
    if pd.isna(val) or str(val).strip() == '':
        return np.nan
    s = str(val).strip().lower().replace(',', '')
    try:
        return float(s)          # digits, with or without thousands separators
    except ValueError:
        pass
    n = text_to_number(s)        # fall back to spelled-out words
    return float(n) if n is not None else np.nan


def standardize_country(val):
    """Collapse the AU / AUS spelling variants onto a single code."""
    if pd.isna(val) or val == '':
        return np.nan
    val = str(val).upper().strip()
    if val in ['AU', 'AUS']:
        return 'AUS'
    return val
