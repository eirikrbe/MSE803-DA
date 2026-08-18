# W4Act1 — Age / Salary / Net-worth Regression & Imputation

MSE803 Data Analysis, Week 4 Activity 1. Cleans a small, deliberately messy dataset
(10 raw rows → 9 people), compares linear vs. polynomial regression, and uses those
models to impute missing values — reporting honestly on how little 9 rows can support.

## Project layout

| Folder | What lives there |
|---|---|
| `data/raw/` | Original input data, treated as read-only — never written to by the notebook. |
| `data/processed/` | Cleaned data produced by the cleaning code alone; no model output. |
| `models/` | Model output — the imputed values, with the model and confidence that produced each. |
| `notebooks/` | The analysis notebook — cleaning, modelling, validation and write-up, in run order. |
| `src/` | Reusable, importable functions the notebook calls; where logic goes once it stabilises. |
| `tests/` | Unit tests for `src/`, runnable without executing the notebook. |
| `figures/` | Every chart the notebook saves; all are regenerated on a full re-run. |
| `scripts/` | Standalone reference code supplied with the assignment; not imported by anything. |

### Which output file to use

A full run writes three files, and the distinction matters:

| File | Contents | Use it when |
|---|---|---|
| `data/processed/W4Act1_cleaned.csv` | 9 people, **observed values only**, gaps left empty | **Default.** Any descriptive statistic or downstream model. |
| `models/predictions.csv` | The 2 imputed cells, with model and confidence per row | You want to know what was estimated and how much to trust it. |
| `data/processed/W4Act1_cleaned_imputed.csv` | The two joined, plus boolean confidence flags | You need one dense table *and* have read the flags. |

They are separate because they have different provenance. The cleaned file is a
deterministic function of the raw CSV and the cleaning code — change a model and it does
not move. The predictions depend on a fitted model, a chosen predictor and a seed.

It also removes a footgun: in the joined file, Alice's observed `Age = 25` and Heidi's
imputed `Age = 32.0` sit in the same column, separated only by a flag you have to know to
filter on. Calling `df['Age'].mean()` on it silently folds in a value produced by a model
with R² = 0.0000. The observed-only file cannot do that to you.

### Why code lives in three places

- **`notebooks/`** — exploration and narrative: prose, code and inline charts read top to
  bottom as a report. Not importable, not tested.
- **`src/`** — logic that has stopped changing, moved out of the notebook so it can be
  imported and unit-tested. The notebook imports from here rather than redefining.
- **`scripts/`** — a third category: a standalone reference script provided with the
  assignment. Neither a module nor part of the pipeline; kept as supplied.

## Running it

The notebook reads and writes via paths relative to `notebooks/`, so run it from there
(or with a tool that sets the notebook's own directory as the working directory):

```bash
data_env/bin/python3 -m nbconvert --to notebook --execute --inplace \
    W4/W4Act1/notebooks/W4Act1_statistics.ipynb
```

A full run regenerates all 9 figures in `figures/` and rewrites
`data/processed/W4Act1_cleaned_imputed.csv`. Nothing in `data/raw/` is modified. All
figure output is deterministic — two consecutive runs produce byte-identical PNGs.

Tests cover `src/` and run in about a second, without touching the notebook:

```bash
data_env/bin/python3 W4/W4Act1/tests/test_cleaning.py
```

They use plain asserts, so they need no extra dependency, but are written to be picked
up by `pytest` if it is ever installed.

## What the analysis covers

- **Cleaning** — text-to-number parsing, comma-separated numerics, country
  standardisation, date parsing, and merging one person split across two rows.
- **Modelling** — univariate linear vs. degree-2 polynomial fits, judged on adjusted R²
  and leave-one-out CV rather than training R² alone.
- **Imputation** — one genuine regression-imputed value, one deliberately weak
  demonstration model, and every filled cell tagged with its confidence.
- **Validation** — held-out testing, exhaustive masking validation, and a permutation
  test of whether any result is statistically decisive.

Findings, caveats and known limitations are written up in the notebook itself; the
Limitations section at the end is the short version.
