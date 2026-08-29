# {{PROJECT_NAME}}

_One or two sentences: what was analysed, and what the answer is._

- **Code:** [`notebooks/`](../notebooks/) — _name the notebooks this project actually used_
- **Data:** [`data/raw/<file>`](../data/raw/) → [`data/processed/<file>`](../data/processed/)
- **Source:** _where the data came from, period covered, what one row is_

## The question

_What this set out to answer, and who acts on the answer._

## 1. The data

n = _<rows>_, _<columns>_ columns. _<What is missing, what was cleaned, what was excluded
and on what basis.>_

| | Variable A | Variable B |
|---|---|---|
| Mean | | |
| Std dev | | |
| Range | | |
| Missing | | |

![Caption stating the finding, not the chart type](../outputs/figures/<name>.png)

## 2. Findings

_One subsection per claim. Each claim gets the number that supports it._

**_<Claim in a sentence.>_** _<Evidence: the statistic, the effect size, n. Then what it
means for the question.>_

![Caption stating the finding](../outputs/figures/<name>.png)

## 3. What this does not show

_The causal reading you are not making, and why. The alternative explanations that remain
open. This section is what separates a result from a claim._

## 4. Method

_What you did and why — enough for someone to reproduce it. Reference the decision log in
`CHECKLIST.md` rather than repeating it._

| Decision | Choice | Reason |
|---|---|---|
| Missingness handling | | |
| Statistical test | | |
| Model (if any) | | |

## 5. Limitations

- **Sample:** _<n, and what it does and does not represent>_
- **Assumptions:** _<linearity, independence, stability over time>_
- **Validation:** _<how the out-of-sample estimate was produced, and its weakness>_
- **Would change with:** _<more data, a control group, a different period>_

---

<!--
Before submitting:
  - every claim above traces to a number in a notebook
  - every figure caption states a finding, not a chart type
  - negative results are here, not buried
  - the decision log in CHECKLIST.md has no blank rows
  - run the humanizer skill over this file (it is calibrated to your voice)
-->
