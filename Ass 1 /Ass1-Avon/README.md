# Ass1-Avon: evidence for Assessment 1, Task 1 (Avon River)

This project analyses the organisation's monitoring dataset for Task 1: three sites, October to December 2023. It
establishes what the data can and cannot support before the written answers rely on it. It covers data
cleaning, derived measures, pre-stated tests and the prototype figures for Task 1-C.

- **Data:** `data/raw/Data_Set_Assignmnet_1 - 0426.xlsx`, from the course platform. It has two tables side by side
  on one sheet (see notebook `01`).
- **Notebooks:**
  - `01_data_understanding`: layout, quality checks and the cleaning DECISIONs, cited by Excel row.
  - `02_eda`: cleaning log, saturation, bands, summaries and Figures A–D.
  - `03_analysis`: hypotheses H1–H6 (Holm-adjusted), a permutation re-check, a sensitivity table across the
    cleaning judgement calls, and the model-versus-baseline check.
  - `04_modeling` is unused, because DECISION 8 is "no".
- **Outputs:** `outputs/figures/` (A temperature, B oxygen saturation, C river-check grid, D īnanga and temperature,
  plus the colour-blindness check), `outputs/tables/` (cleaning log, summaries, tests, sensitivity) and `data/processed/`
  (`visits.csv`, `fish_records.csv`).
- **Key results:**
  - 100 → 69 visits and 90 → 70 fish records after cleaning. Every row below Excel row 73 is a copy, exact
    or altered, or an invalid record.
  - The river warmed about 3 °C (Oct → Dec), with AV-3 warmest.
  - All 69 visits were below saturation, and 12 fell under 7 mg/L.
  - Īnanga counts per survey halved in December (Holm p = 0.03; permutation p = 0.002).
  - No count ~ water-quality association was found, and no model beats the mean.
  - No alternative cleaning decision changes any of these conclusions (sensitivity table in `03`).
- **Status:** every DECISION was confirmed on 3 Oct 2026. The decision log is at the bottom of `CHECKLIST.md`.

> The MSE803-DA repository is public. Keep this folder out of any push until the assessment has been graded.

## Running it

```bash
# from the MSE803-DA repo root (openpyxl is needed to read the .xlsx)
data_env/bin/jupyter nbconvert --to notebook --execute --inplace \
    "Ass 1 /Ass1-Avon/notebooks/Ass1-Avon_01_data_understanding.ipynb" \
    "Ass 1 /Ass1-Avon/notebooks/Ass1-Avon_02_eda.ipynb" \
    "Ass 1 /Ass1-Avon/notebooks/Ass1-Avon_03_analysis.ipynb"
```

Built from `MSE803-Analytics-Template`. `CHECKLIST.md` describes the workflow.
