# W4Act2 — World Happiness Dashboard

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue)]()

Ranks 20 countries by happiness, compares the lowest-scoring country's freedom score to
the sample average, checks which factors correlate with happiness, and screens every
column for outliers with both the IQR and Z-score methods. Answer: nothing correlates
strongly, and neither method flags any value as an outlier at standard thresholds.

- **Data:** `data/raw/world_happiness_dataset.csv` — source/period not stated by the file
  itself; one row is one country
- **Code:** `notebooks/01_data_understanding.ipynb`, `notebooks/02_eda.ipynb`
- **Findings:** [`reports/W4Act2.md`](reports/W4Act2.md) · **Figures:** `outputs/figures/`
  · **Dashboards:** `outputs/dashboards/`

## Running it

```bash
python3 -m venv data_env
data_env/bin/pip install -r requirements.txt
data_env/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

## Layout

```
data/raw/         original CSV, never edited in place
notebooks/        01_data_understanding, 02_eda
src/              loading, quality checks, cleaning, correlation, plotting, dashboards
outputs/          figures/ tables/ dashboards/
reports/          the write-up
tests/            standalone-runnable checks for src/data/{quality,cleaning}.py
```

Notebooks call into `src/` rather than repeating chart or cleaning code inline:

| Module | What it does |
|---|---|
| `data/loading.py` | `load_raw`, `list_raw` |
| `data/quality.py` | `profile`, `schema_report`, `missing_report`, `duplicate_report`, `cardinality_report`, `outlier_report` (IQR + Z-score) |
| `data/cleaning.py` | `snake_case_columns`, `strip_whitespace` |
| `analysis/descriptive.py` | `describe_numeric`, `describe_categorical` |
| `analysis/correlation.py` | `correlation_matrix`, `top_correlations_with` |
| `visualization/theme.py` | `PALETTE`, `set_plot_style`, `save_fig`, `highlight`, `diverging` |
| `visualization/plots.py` | `distribution_grid`, `correlation_heatmap`, `ranked_bar`, `correlation_bars` |
| `visualization/presentation.py` | `build_dashboard`, `save_table` |

## Tests

```bash
data_env/bin/python3 tests/test_cleaning.py
data_env/bin/python3 tests/test_quality.py
```

Plain functions with plain asserts and a standalone runner (no `pytest` required), run
against synthetic data with a known ground truth rather than this project's own dataset.
