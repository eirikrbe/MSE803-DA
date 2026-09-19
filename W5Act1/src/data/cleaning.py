"""Cleaning helpers that generalise across datasets.

clean_number and text_to_number are promoted from W3/W3Act1-2/src/cleaning.py,
where they were written and tested. standardize_country became the generic
standardize_category.

These are TOOLS, not a pipeline. Which of them a dataset needs, in what order,
and what counts as clean, is dataset-specific and stays in the notebook.
"""

import re

import numpy as np
import pandas as pd

# Spelled-out number vocabulary. Deliberately small - a wider vocabulary would
# need a real parser, not a lookup.
word_to_num = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
    'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
    'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
    'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
    'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90, 'hundred': 100,
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
    if 'million' in tokens:
        total *= 1_000_000
    return total if total > 0 else None


def clean_number(val):
    """Parse a numeric cell however it was written: 38, '30,000', '$1,200', 'thirty-eight'.

    ONE parser for every numeric column. An earlier version in W3 used a
    per-column lookup, which silently returned NaN for every spelled-out value it
    had not been told about - a bug the dataset happened to hide. Keep it general.
    """
    if pd.isna(val) or str(val).strip() == '':
        return np.nan
    s = str(val).strip().lower().replace(',', '').replace('$', '').replace('%', '')
    try:
        return float(s)
    except ValueError:
        pass
    n = text_to_number(s)
    return float(n) if n is not None else np.nan


def snake_case_columns(df):
    """'Happiness Score' -> 'happiness_score'. Returns a copy.

    Purely mechanical: consistent column names mean you stop guessing whether it
    was `GDP_per_Capita` or `gdp per capita` on every line you write.
    """
    def convert(c):
        s = str(c)
        # Two camelCase boundaries, both needed: 'netWorth' -> 'net_Worth', and
        # 'GDPPerCapita' -> 'GDP_PerCapita' (an acronym running into a word).
        s = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', s)
        s = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', '_', s)
        s = re.sub(r'[^0-9a-zA-Z]+', '_', s).lower()
        return re.sub(r'_+', '_', s).strip('_')

    out = df.copy()
    out.columns = [convert(c) for c in out.columns]
    return out


def strip_whitespace(df):
    """Trim leading/trailing whitespace in every object column. Returns a copy."""
    out = df.copy()
    for col in out.select_dtypes(include='object').columns:
        out[col] = out[col].map(lambda v: v.strip() if isinstance(v, str) else v)
    return out


def standardize_category(val, aliases, case='upper'):
    """Collapse spelling variants onto one label using an explicit alias map.

    aliases maps variant -> canonical, e.g. {'AU': 'AUS', 'AUSTRALIA': 'AUS'}.
    The map is a required argument because which variants mean the same thing is
    a fact about your data that only you know. Unmapped values pass through
    normalised but unchanged, so nothing is silently reclassified.
    """
    if pd.isna(val) or str(val).strip() == '':
        return np.nan
    s = str(val).strip()
    s = s.upper() if case == 'upper' else s.lower() if case == 'lower' else s
    lookup = {(k.upper() if case == 'upper' else k.lower() if case == 'lower' else k): v
              for k, v in aliases.items()}
    return lookup.get(s, s)


def coerce_types(df, spec):
    """Apply an explicit {column: type} map. 'numeric' routes through clean_number.

    spec is required and never inferred: pandas guessing a dtype is how a postcode
    becomes an integer and loses its leading zero.
    """
    out = df.copy()
    for col, kind in spec.items():
        if col not in out.columns:
            raise KeyError(f'{col!r} is not a column. Have: {list(out.columns)}')
        if kind == 'numeric':
            out[col] = out[col].map(clean_number)
        elif kind == 'datetime':
            out[col] = pd.to_datetime(out[col], errors='coerce')
        elif kind == 'category':
            out[col] = out[col].astype('category')
        else:
            out[col] = out[col].astype(kind)
    return out


def drop_exact_duplicates(df, subset=None, keep='first'):
    """Drop exact duplicate rows and SAY how many went. Returns a copy.

    The only function in this package that removes rows, and it prints what it
    did. Anything less visible than that belongs in your notebook where the
    reasoning is written down next to it.
    """
    before = len(df)
    out = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    removed = before - len(out)
    print(f'dropped {removed} duplicate row(s); {len(out)} remain'
          + (f' (subset={subset})' if subset else ''))
    return out
