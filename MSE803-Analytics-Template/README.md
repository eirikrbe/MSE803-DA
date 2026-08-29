# {{PROJECT_NAME}}

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue)]()


_One sentence: what this analyses and what question it answers._

- **Data:** `data/raw/<file>` — _source, period, what one row is_
- **Notebooks:** `notebooks/` — _which stages this project actually used_
- **Findings:** `reports/` · **Figures:** `outputs/figures/`

---

<!-- Everything below is template documentation. Delete it once the project is real. -->

## What this template is

A reusable **workflow and engineering** scaffold for a data analytics project. It
standardises the process and the plumbing so you can start on a new dataset without
rebuilding the same structure every time.

**It provides:** project structure, data loading, schema inspection, data-quality checks,
missing-value analysis, duplicate detection, profiling, visualisation utilities,
reproducibility and testing.

**It does not provide:** the analytical question, the transformations, the choice of
statistical method, the features, the model, or the conclusions. Those are specific to
every dataset and every question, and they stay yours.

> **Automate the mechanics; expose the judgement.**

The framework will happily tell you that a column is 40% missing. It will not decide
whether to impute it — `impute()` has no default `method=` and raises until you choose
one. The same is true of statistical tests, predictor sets, model selection and outlier
removal. Those appear as `## DECISION` cells in the notebooks and in the decision log at
the bottom of `CHECKLIST.md`.

## Starting a new project

```bash
# From the MSE803-DA repo root
./new_project.sh W5Act1 ~/Documents/GitHub/MSE803-DA/W5
```

Or copy the folder by hand — `cp -R MSE803-Analytics-Template W5/W5Act1` — and rename what
you need.

Then:

1. Put the original file in `data/raw/` and **never edit it in place**.
2. Set `DATASET = 'yourfile.csv'` in the first cell of each notebook you use.
3. Open `01_data_understanding.ipynb` and answer DECISION 1 before running anything.
4. Work down `CHECKLIST.md`.
5. Delete the notebooks your question does not need.

## The workflow

```
Business question → Data acquisition → Data understanding → Data validation →
Data cleaning → EDA → Transformation (when justified) → Statistical analysis →
Modelling (when justified) → Visualisation → Communication →
Deployment (when appropriate) → Automation (when appropriate)
```

The process stays constant. The implementation changes with the dataset. Stages marked
"when justified" are genuinely optional — see `CHECKLIST.md`.

## Notebooks are stages, not mandatory steps

| Notebook | Stage |
|---|---|
| `01_data_understanding.ipynb` | Acquisition, understanding, validation |
| `02_eda.ipynb` | Cleaning, exploratory analysis |
| `03_analysis.ipynb` | Transformation, statistical analysis |
| `04_modeling.ipynb` | Modelling — **optional** |

Use the ones your question needs and delete the rest:

- A descriptive project may stop after `03` and never open `04`. That is a finished
  analysis, not an incomplete one.
- A modelling project may need feature-engineering work that fits none of the four and
  earns its own notebook.
- A messy dataset may spend most of its life inside `01` before `02` means anything.
- A small, clean, well-understood dataset may only need `02`.

## Notebooks vs. `src/`

| Notebooks are for | `src/*.py` is for |
|---|---|
| Investigation and exploration | Reusable logic |
| Experimentation | Data loading and validation utilities |
| Visual reasoning | Cleaning functions that generalise |
| Documenting analytical decisions | Analysis, visualisation, modelling utilities |
| Presenting results | Anything worth a test |

Notebooks **call** `src/` rather than growing into standalone scripts. The migration rule:
once a piece of logic stops changing and gets used twice, move it to `src/` and give it a
test. Logic specific to one dataset's quirks legitimately stays in the notebook.

## What is in `src/`

| Module | What it does |
|---|---|
| `paths.py` | Every path, resolved from the project root. Nothing depends on the working directory |
| `data/loading.py` | `load_raw`, `save_interim`, `save_processed`, `list_raw` |
| `data/quality.py` | `profile`, `schema_report`, `missing_report`, `duplicate_report`, `outlier_report`, `cardinality_report`, `quality_report`. **All report; none mutate or drop** |
| `data/cleaning.py` | `clean_number`, `text_to_number`, `snake_case_columns`, `standardize_category`, `coerce_types`, `strip_whitespace`, `drop_exact_duplicates` |
| `analysis/descriptive.py` | `describe_numeric`, `describe_categorical`, `group_compare` |
| `analysis/correlation.py` | `correlation_matrix`, `top_correlations_with` |
| `analysis/hypothesis.py` | `compare_groups` (requires `test=`), `chi2_independence`, `permutation_test`, `cohens_d` |
| `visualization/theme.py` | `PALETTE`, `set_plot_style`, `save_fig`, `highlight` |
| `visualization/plots.py` | `missingness_matrix`, `correlation_heatmap`, `distribution_grid`, `ranked_bar`, `correlation_bars`, `category_counts`, `scatter_with_fit` |
| `visualization/presentation.py` | `build_dashboard`, `plotly_dashboard`, `save_table`, `export_for_bi` |
| `models/baseline.py` | Univariate regression + LOOCV. `compare_models` reports and never picks a winner |
| `models/imputation.py` | `impute` (requires `method=`), `write_provenance_artifacts` |

## Presentation layer

A dashboard is not always the right output. Pick from the audience:

| Audience | Output | How |
|---|---|---|
| Assessment, peer review | Executed notebook + `reports/*.md` | The default |
| A report figure | Static PNG | `save_fig()` |
| Reader explores values | Plotly HTML | `plotly_dashboard()` |
| Power BI / Tableau user | Flat typed extract | `export_for_bi()` |
| Interactive tool, real users | Streamlit | `pip install streamlit`, add an `app.py`. Not a default dependency — add it when a real user needs it |

## Setup

```bash
python3 -m venv data_env
data_env/bin/pip install -r requirements.txt
```

Or reuse the shared `data_env` at the MSE803-DA repo root.

## Tests

`pytest` is not installed in `data_env`, so the tests are plain functions with plain
asserts and a standalone runner:

```bash
data_env/bin/python3 tests/test_cleaning.py
data_env/bin/python3 tests/test_quality.py
data_env/bin/python3 tests/test_models.py
```

They also collect under `pytest` if you install it. They test against synthetic data with
a known ground truth, never a project's own dataset, which is what keeps them portable
when the template is copied.

## The two assistants

`.claude/agents/` defines two subagents:

- **`data-analyst`** — implements *your chosen* method. Never invents a methodological
  decision because a default was available.
- **`analysis-reviewer`** — audits the analysis: unsupported claims, training R² cited as
  evidence, leakage, causal language, and unexamined decisions. Challenges whether a
  choice was justified; never makes the choice.

Ask for them by name: *"use analysis-reviewer on 03_analysis"*.
