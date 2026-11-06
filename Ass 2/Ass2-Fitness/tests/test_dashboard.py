"""Tests for src/visualization/dashboard.py: the published page stays self-contained and
aggregate-only. Synthetic inputs only, so the tests survive being copied to another project.

Run standalone:  data_env/bin/python3 tests/test_dashboard.py
"""

import re
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.paths import DASHBOARDS
from src.visualization.dashboard import MIN_CELL, TEMPLATE, suppress_small


def test_template_is_one_self_contained_page():
    html = TEMPLATE.read_text(encoding='utf-8')
    assert html.lower().startswith('<!doctype html>')
    assert 'name="viewport"' in html, 'needs a viewport tag for phones'
    assert not re.search(r'<script[^>]+src=', html), 'must load no script from outside the file'
    assert html.count('__DATA__') == 1, 'exactly one data slot'


def test_small_counts_are_suppressed_and_zeros_kept():
    assert suppress_small([[0, 1, 4, 5, 12]], min_cell=5) == [[0, None, None, 5, 12]]


def test_threshold_is_at_least_five():
    assert MIN_CELL >= 5, 'publishing groups smaller than five users breaks guideline G6'


def test_published_dashboard_does_not_include_record_ids():
    html = (DASHBOARDS / 'fitness_dashboard.html').read_text(encoding='utf-8')
    assert not re.search(r'\buser\s+\d{3,}\b|\bIDs?\s+\d{3,}\b', html, re.IGNORECASE)


if __name__ == '__main__':
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith('test_') and callable(f)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f'  PASS  {name}')
        except AssertionError as exc:
            failures += 1
            print(f'  FAIL  {name}: {exc}')
    print(f'\n{len(tests) - failures}/{len(tests)} passed')
    sys.exit(1 if failures else 0)
