# FairGuard AI — User Manual

> **Audience:** anyone who wants to *run* FairGuard by hand — no IDE, no wizard.
> **Scope:** every command in this document is copy-pasteable from the repo root and
> was executed on this machine (Windows 11, Python 3.12.6).

**Related documents**

| Document | What it is |
| --- | --- |
| [`EXPLAIN.md`](EXPLAIN.md) | What the project *is*, how every module works, and the data flow |
| [`ROOT_CAUSES.md`](ROOT_CAUSES.md) | The 24 root causes found in the audit and how each was fixed |
| [`README.md`](README.md) | Short overview + quickstart |

---

## Table of contents

1. [What FairGuard does](#1-what-fairguard-does)
2. [Install](#2-install)
3. [The manual flow in one picture](#3-the-manual-flow-in-one-picture)
4. [Step 1 — Create the demo data](#step-1--create-the-demo-data)
5. [Step 2 — Train a model](#step-2--train-a-model)
6. [Step 3 — Run an assessment](#step-3--run-an-assessment)
7. [Step 4 — Read the report](#step-4--read-the-report)
8. [Step 5 — Drift against a real reference](#step-5--drift-against-a-real-reference)
9. [Step 6 — Compare in-sample vs out-of-sample](#step-6--compare-in-sample-vs-out-of-sample)
10. [Step 7 — Check text / LLM outputs](#step-7--check-text--llm-outputs)
11. [Step 8 — Assess a real dataset](#step-8--assess-a-real-dataset)
12. [Step 9 — Use the REST API](#step-9--use-the-rest-api)
13. [Step 10 — Use the Streamlit dashboard](#step-10--use-the-streamlit-dashboard)
14. [Step 11 — Use the React frontend](#step-11--use-the-react-frontend)
15. [Step 12 — Run the tests](#step-12--run-the-tests)
16. [Step 13 — Inspect the MLflow history](#step-13--inspect-the-mlflow-history)
17. [Reference — every CLI flag](#reference--every-cli-flag)
18. [Reference — the API endpoints](#reference--the-api-endpoints)
19. [Reference — configuration knobs](#reference--configuration-knobs)
20. [Troubleshooting](#20-troubleshooting)
21. [Command cheat sheet](#21-command-cheat-sheet)

---

## 1. What FairGuard does

FairGuard takes **one trained model + one dataset** and produces **one assessment record**
that answers six questions:

| Dimension | Question | Key numbers |
| --- | --- | --- |
| **Performance** | Is the model any good, out of sample? | accuracy, precision, recall, F1, ROC-AUC, Brier, calibration error (ECE) |
| **Fairness** | Does it treat groups differently? | demographic-parity difference, equalized-odds difference, per-group metrics, 95 % bootstrap CIs |
| **Explainability** | *Why* did it decide that? | mean \|SHAP\| per feature (global), signed per-row contributions (local) |
| **Privacy** | Is personal data leaking in? | PII-like columns, severity, masked samples |
| **Drift** | Has the input distribution moved? | PSI + level, KS p-value, per feature |
| **Text safety** | Are generated outputs safe? | toxicity / bias / hate-speech / PII scores per text |

All of it is combined into a single **risk score (0–1) → Low / Medium / High / Critical**
plus a **suggested decision** (Approve → Approve with monitoring → Needs review → Block)
and plain-language **recommendations**.

Everything runs **locally**. No paid APIs, no accounts, no telemetry.

---

## 2. Install

```powershell
# from the repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate         # macOS / Linux

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional — the transformer-based text judges (toxicity, bias, hate speech, PII,
hallucination). Without them FairGuard falls back to its built-in offline
heuristics, which still work but are cruder:

```powershell
python -m pip install transformers torch
```

Check the install:

```powershell
python -c "import sklearn, shap, fairlearn, pandas, numpy; print('ok', sklearn.__version__, shap.__version__, fairlearn.__version__)"
```

> **Windows one-click start:** the repo also ships `start.bat`, which generates
> the demo data and models if they are missing and then launches the FastAPI
> backend (http://localhost:8000), the Streamlit dashboard (http://localhost:8501)
> and, when `frontend/node_modules` exists, the React frontend
> (http://localhost:5173). `stop.bat` kills all three. Everything below shows
> what those steps actually do, one command at a time.

---

## 3. The manual flow in one picture

```
                 ┌──────────────────────────┐
                 │  data/<your>.csv         │
                 └────────────┬─────────────┘
                              │
             scripts/train_demo_model.py      ← trains, keeps a hold-out split
                              │
                 ┌────────────▼─────────────┐
                 │  models/<name>.pkl       │  (model + features + holdout + meta)
                 └────────────┬─────────────┘
                              │
            scripts/run_assessment.py        ← scores the HOLDOUT rows
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
  performance              fairness            explainability
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
              privacy ── drift ── text safety
                              │
                  fairguard/risk.py  → score + level + decision
                              │
                 ┌────────────▼──────────────┐
                 │ reports/assessment_*.json │
                 │ reports/assessment_*.md   │
                 │ reports/model_card_*.md   │
                 │ mlruns/mlflow.db          │
                 └───────────────────────────┘
```

> **Two rules that matter**
> 1. **Drift needs a second dataset.** `--reference` must point at a *different*
>    file. Comparing a dataset with itself always yields PSI = 0 — that is a
>    baseline, not evidence of stability.
> 2. **Assessments score the hold-out** by default. Rows the model was fitted on
>    are not evidence that the model works.

---

## Step 1 — Create the demo data

```powershell
python scripts/make_demo_data.py
```

Expected output:

```
wrote C:\...\FairGuard\data\demo_loans.csv (2000 rows)
wrote C:\...\FairGuard\data\demo_loans_new.csv (2000 rows) [drifted]
```

| File | Purpose |
| --- | --- |
| `data/demo_loans.csv` | The assessed dataset (synthetic loan applications). |
| `data/demo_loans_new.csv` | A **drifted** copy — same schema, shifted distributions. Use it as the drift reference. |

---

## Step 2 — Train a model

```powershell
python scripts/train_demo_model.py
```

Expected output (measured on this machine):

```
logistic_regression   hold-out accuracy=0.948  (n_test=500, stratified=True)  -> models\logistic_regression.pkl
random_forest         hold-out accuracy=0.990  (n_test=500, stratified=True)  -> models\random_forest.pkl
gradient_boosting     hold-out accuracy=0.992  (n_test=500, stratified=True)  -> models\gradient_boosting.pkl
```

Useful variations:

```powershell
# only one model
python scripts/train_demo_model.py --model-type logistic_regression

# a different hold-out size (default 0.25)
python scripts/train_demo_model.py --test-size 0.3

# turn off balanced class weights (rarely a good idea, useful for comparison)
python scripts/train_demo_model.py --no-balance
```

What gets written to `models/*.pkl`:

| Key | Meaning |
| --- | --- |
| `model` | the fitted sklearn `Pipeline` |
| `features` | the encoded column names the model expects |
| `raw_features` | the pre-encoding column names (used for drift folding) |
| `model_type` | `logistic_regression` / `random_forest` / `gradient_boosting` |
| `holdout` | the hold-out split: `test_index`, `n_test`, `stratified`, `test_size` |
| `meta` | label, whether class balancing was on, the hold-out accuracy |

---

## Step 3 — Run an assessment

```powershell
python scripts/run_assessment.py --sensitive gender
```

Expected output (measured):

```
Evaluation scope: holdout (500 rows scored)
Risk score: 0.5298 -> High (suggested decision: Needs review)
Risk coverage: 65% (dimensions: fairness, performance, privacy)

Recommendations:
  - Fairness: ...
  - Performance: ...
  - Privacy: PII-like columns found (email, full_name, phone) - ...

Saved:
  json: C:\...\reports\assessment_20261022T141237.json
  markdown: C:\...\reports\assessment_20261022T141237.md
  model_card: C:\...\reports\model_card_20261022T141237.md
```

Read the three lines at the top carefully:

* **Evaluation scope** — `holdout` means out-of-sample (good). `full_dataset`
  means the model was trained on part of what you are looking at.
* **Risk score → level** — the headline number.
* **Risk coverage** — how much of the total risk weight was actually assessed
  (65 % here because no drift reference and no texts were supplied). Scores are
  only comparable between assessments with the **same coverage**.

---

## Step 4 — Read the report

Open the Markdown file:

```powershell
notepad reports\assessment_20261022T141237.md
```

or print it in the terminal:

```powershell
Get-Content reports\assessment_20261022T141237.md
```

The report contains, in order:

1. **Evaluation scope** — hold-out vs full dataset, rows scored.
2. **Risk summary** — score, level, suggested decision, dimensions assessed,
   weight coverage, and the per-dimension contribution table.
3. **Performance** — accuracy, precision, recall, F1, ROC-AUC, Brier, ECE.
4. **Fairness** — DPD and EOD *with* 95 % confidence intervals, the per-group
   table (rows, selection rate, accuracy, precision, recall), and a warning
   when a group is too small to trust.
5. **Explainability** — which SHAP explainer ran, the background size, and the
   top features by mean \|SHAP\|.
6. **Privacy findings** — column, type, rows affected, severity.
7. **Drift** — PSI + level and KS p-value per feature, plus a note saying
   which reference dataset was used.
8. **Recommendations** — plain-language actions.
9. **Reviewer decision** — where a human sign-off lands.

The machine-readable version is the JSON file next to it:

```powershell
# just the risk block
(Get-Content reports\assessment_20261022T141237.json -Raw | ConvertFrom-Json).risk | Format-List

# just the fairness block
(Get-Content reports\assessment_20261022T141237.json -Raw | ConvertFrom-Json).fairness | Format-List
```

---

## Step 5 — Drift against a real reference

Drift is only meaningful against a **different** dataset:

```powershell
python scripts/run_assessment.py --sensitive gender --reference data/demo_loans_new.csv --model models/gradient_boosting.pkl
```

Expected output (measured):

```
Evaluation scope: holdout (500 rows scored)
Risk score: 0.7335 -> High (suggested decision: Needs review)
Risk coverage: 85% (dimensions: drift, fairness, performance, privacy)

Recommendations:
  - Fairness: demographic parity difference ... and equalized odds difference 0.38 exceed the configured limits ...
  - Privacy: PII-like columns found (email, full_name, phone) - ...
  - Drift: major distribution shift in income_lakhs - re-validate on recent data and schedule retraining.
```

What to look for:

* `Risk coverage` jumped from 65 % → 85 % — the drift dimension now contributes.
* The Drift section of the report lists per-feature PSI. `PSI < 0.10` = stable,
  `0.10–0.25` = moderate, `> 0.25` = major.

If you accidentally point `--reference` at the assessed dataset, FairGuard warns
you and the report marks it as a baseline:

```
warning: --reference is the same dataset as --data — every PSI would be 0 by construction.
```

---

## Step 6 — Compare in-sample vs out-of-sample

This is the fastest way to see why the hold-out matters:

```powershell
# out-of-sample (default)
python scripts/run_assessment.py --sensitive gender --model models/logistic_regression.pkl

# in-sample (everything, including the training rows)
python scripts/run_assessment.py --sensitive gender --model models/logistic_regression.pkl --full-dataset
```

| Run | Scope | Rows scored | Risk |
| --- | --- | --- | --- |
| default | `holdout` | 500 | 0.5298 → High |
| `--full-dataset` | `full_dataset` | 2000 | *see your run* |

The two reports differ in the **Evaluation scope** section and in the accuracy
number. The in-sample accuracy is optimistic — that is exactly the illusion the
audit removed.

---

## Step 7 — Check text / LLM outputs

```powershell
python scripts/run_assessment.py --sensitive gender --texts data/sample_texts.csv
```

The CSV needs a `text` column and, optionally, a `reference` column (needed by
the hallucination judge):

```csv
text,reference
"The service is terrible and the staff are rude.","The service is slow and the staff were busy."
"My card number is 4111111111111111 and my email is a@b.com",""
```

Checks performed per text: **toxicity**, **bias**, **hate speech**, **PII**
(and **hallucination** when a `reference` is supplied).

* Without `transformers` installed → the built-in offline heuristics run and the
  report says so per check (`backends` in the JSON).
* With `transformers` installed → the matching Hugging Face model runs locally.
  The first run downloads the model (~100–400 MB each).

The report shows the backend for **every** check, so a silent fallback to
heuristics can never hide behind another kind's backend.

---

## Step 8 — Assess a real dataset

Nine public datasets ship in `data/`. Any CSV with a binary label works.

```powershell
# 1. download / refresh the public datasets (already present in this repo)
python scripts/download_datasets.py

# 2. train
python scripts/train_demo_model.py --data data/adult.csv --label income_gt_50k --sensitive sex

# 3. assess
python scripts/run_assessment.py --data data/adult.csv --label income_gt_50k --sensitive sex --model models/adult_logistic_regression.pkl
```

| Dataset | Label column | Sensitive column |
| --- | --- | --- |
| `adult.csv` | `income_gt_50k` | `sex` |
| `german.csv` | `credit_good` | `sex` |
| `taiwan_credit.csv` | `default_next_month` | `gender` |
| `compas_two_years.csv` | `two_year_recid` | `race` |
| `bank_marketing.csv` | `subscribed` | `age` |
| `student_performance.csv` | `G3_pass` | `sex` |
| `heart.csv` | `disease` | `sex` |
| `communities.csv` | `high_violent_crime` | `majority_white` |
| `australian.csv` | `credit_approved` | `gender_like` |

> The sensitive column is **kept out of the model features** (the standard
> fair-AI setup) but is still measured at assessment time. Passing
> `--sensitive ""` keeps it in the model — only do that for teaching purposes.

> **Leakage guard:** if a CSV contains a raw copy of the target (e.g.
> `adult.csv` has the string column `income` beside the derived label
> `income_gt_50k`), the training script refuses it (`fairguard.train.
> label_duplicate_columns`). The model can no longer "read the answer":
> Adult's hold-out accuracy went from a nonsensical 1.000 to 0.829 once the
> duplicate was detected and dropped.
>
> The **assessment repeats this check** (defence in depth): training a model
> with an older artifact or a hand-picked feature list that maps 1:1 onto the
> label makes every report headline say
> `## ⚠️ Label leakage detected`, the recommendations open with a
> "Leakage: … retrain" line, and `evaluation.in_sample` is forced `true`.

**Retrain everything at once.** Models saved by older versions of FairGuard
lack the hold-out split / one-hot categoricals / balanced weights / leak
guard. One command refreshes every artifact in `models/`:

```powershell
python scripts/retrain_all.py            # all 10 datasets × 3 model types (≈3 min)
python scripts/retrain_all.py --only adult                       # one dataset
python scripts/retrain_all.py --models logistic_regression       # one model type
```

To see what each dataset looks like before training:

```powershell
python -c "import sys; sys.path.insert(0,'.'); from fairguard import data; import json; print(json.dumps(data.profile_dataset(data.load_csv('data/adult.csv'))['numeric_summary'], indent=2, default=str))"
```

---

## Step 9 — Use the REST API

```powershell
python api.py
```

The API listens on `http://127.0.0.1:8000`. Interactive docs: `http://127.0.0.1:8000/docs`.

### End-to-end example with PowerShell

```powershell
# 0. train (returns the model names + hold-out size)
$train = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/train `
  -ContentType 'application/json' -Body '{"dataset":"demo_loans.csv","label":"approved","sensitive":"gender"}'
$train.results | Format-Table model, test_accuracy, holdout_rows

# 1. assess the hold-out, no drift reference
$body = @{
  dataset   = 'demo_loans.csv'
  model     = 'gradient_boosting.pkl'
  label     = 'approved'
  sensitive = 'gender'
} | ConvertTo-Json
$assess = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/assess `
  -ContentType 'application/json' -Body $body

$assess.run_id                      # -> 20261022T141237
$assess.assessment.evaluation       # -> {scope: holdout, n_rows: 500, in_sample: False}
$assess.assessment.risk              # -> {score: .., level: .., coverage: ..}
$assess.assessment.fairness.demographic_parity_difference_ci

# 2. drift against a different dataset
$body2 = @{
  dataset   = 'demo_loans.csv'
  model     = 'gradient_boosting.pkl'
  label     = 'approved'
  sensitive = 'gender'
  reference = 'demo_loans_new.csv'
} | ConvertTo-Json
$assess2 = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/assess `
  -ContentType 'application/json' -Body $body2
$assess2.assessment.drift_reference.same_as_assessed   # -> False

# 3. record a reviewer decision on THAT run
$review = @{
  run_id   = $assess2.run_id
  decision = 'Approve with monitoring'
  note     = 'Fairness EOD to be re-checked next quarter'
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/review `
  -ContentType 'application/json' -Body $review

# 4. export the report
$md = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/report/$($assess2.run_id)/md"
$md.content | Out-File -Encoding utf8 reports\$($assess2.run_id).md

# 5. history
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/history | Select-Object -ExpandProperty runs |
  Select-Object -First 5 model, level, risk, scope, explainer | Format-Table
```

> **Why `run_id` matters:** the review endpoint writes the decision onto the
> assessment you name. Passing the wrong (or no) id used to overwrite the
> newest report on disk regardless — that bug is fixed and covered by a test.

---

## Step 10 — Use the Streamlit dashboard

```powershell
streamlit run app.py
```

Open `http://127.0.0.1:8501`.

Sidebar workflow:

1. **Dataset** — *Demo dataset* (generated on the fly if the CSV is missing) or
   *Upload CSV*.
2. **Model** — pick any `models/*.pkl`.
3. **Label column** / **Sensitive attribute** — pick from the columns.
4. **Decision threshold** — the cutoff for `approved` (default 0.50).
5. **Drift reference dataset (optional)** — *— none —* by default. Pick a
   **different** CSV to enable the drift check.
6. **Optional: texts to check** — one output per line.
7. Press **Run assessment**.

Tabs:

| Tab | Shows |
| --- | --- |
| Overview | risk score, level, decision, evaluation scope, coverage, risk contributions, recommendations, dataset profile |
| Fairness | DPD / EOD with CIs, per-group metrics, small-group warning |
| Explainability | which SHAP explainer ran, top-10 global importance, single-row local explanation |
| Privacy | PII findings table (masked samples) |
| Drift | PSI per feature with colour-coded level, or an explanation of why drift was skipped |
| Performance | accuracy/precision/recall/F1/ROC-AUC, ECE, Brier, confusion matrix |
| Text & LLM | per-check backend, mean score per check, per-text table |
| Review | save a reviewer decision (it appears in the exported report) |
| History | the MLflow run history |
| Report | download JSON / Markdown / model card, with a live preview |

---

## Step 11 — Use the React frontend

```powershell
cd frontend
npm install
npm run dev
```

The Vite dev server prints a URL (usually `http://localhost:5173`). It talks to
the FastAPI backend, so start that first:

```powershell
# terminal 1
python api.py
# terminal 2
cd frontend && npm run dev
```

| Page | What it does |
| --- | --- |
| **Dashboard** | headline risk numbers, the run history, quick links |
| **Train** | pick a dataset + model types, train, see the hold-out accuracy |
| **Assess** | pick dataset, model, sensitive attribute, optional drift reference, run |
| **Results** | the full result set — performance, fairness, explainability, privacy, drift, recommendations |
| **History** | every logged assessment |

Production build:

```powershell
cd frontend
npm run build      # -> frontend/dist/
```

---

## Step 12 — Visualize every dataset

Before (or instead of) running models, see the data itself. This writes 99
PNG charts into `visualize/` — one folder per dataset:

```powershell
python scripts/visualize_data.py                 # every dataset (~1 minute)
python scripts/visualize_data.py --only adult    # a single dataset
```

Then open `visualize/README.md`, which explains how to read each one. The
suite per dataset:

| Chart | What it shows |
| --- | --- |
| `01_label_distribution.png` | how balanced the target is (a 95/5 target makes plain accuracy meaningless) |
| `02_numeric_distributions.png` | histogram grid of the top numeric features (ordered by label correlation) |
| `03_categorical_counts.png` | counts of low-cardinality categorical columns |
| `04_label_correlation.png` | which features raise/lower the label — a **sensitive** column high here predicts a fairness finding (e.g. `gender_like` ≈ 0.45 on Australian credit) |
| `05_correlation_heatmap.png` | which features duplicate each other; a feature 1:1 with the label is the RC-23 leak the pipeline refuses |
| `06_missing_values.png` | % missing per column |
| `07_group_selection_rate.png` | P(label = 1) per sensitive group — the fairness lens before any model is trained |
| `08_boxplots_top_features.png` | feature spread split by label class (outliers hidden) |
| `09_scatter_top_features.png` | the two strongest features against each other, coloured by class |

Two extras: `visualize/drift/10_drift_demo_loans.png` (PSI per feature
between the demo dataset and its drifted copy) and
`visualize/text/11_text_safety.png` (heuristic judge scores for the sample
texts).

---

## Step 13 — Run the tests

```powershell
python -m pytest tests -q
```

Expected:

```
125 passed in 15s
```

Single file / single test:

```powershell
python -m pytest tests/test_fairness_ci.py -v
python -m pytest tests/test_holdout_scope.py::test_holdout_matches_training_split -v
```

Coverage-ish view of what is protected:

```powershell
python -m pytest tests -q --collect-only | Select-String "test_" | Measure-Object
```

The same command runs in CI: `.github/workflows/tests.yml` (push + pull request,
Python 3.12, pip cache).

---

## Step 14 — Inspect the MLflow history

Every assessment is logged as an MLflow run (SQLite, no server needed):

```powershell
python -c "import sys; sys.path.insert(0,'.'); from fairguard import tracking; print(tracking.history().to_string())"
```

Columns include `model`, `version`, `sensitive`, `risk`, `level`, `decision`,
`reviewer`, **`scope`** (holdout / full_dataset) and **`explainer`**
(linear / tree / permutation).

To browse it in a UI:

```powershell
python -m pip install mlflow
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
# then open http://127.0.0.1:5000
```

---

## Reference — every CLI flag

### `scripts/train_demo_model.py`

| Flag | Default | Meaning |
| --- | --- | --- |
| `--data` | *(demo)* | CSV to train on |
| `--label` | `approved` | target column |
| `--sensitive` | *(none)* | sensitive column to exclude from the features |
| `--model-type` | *(all)* | `logistic_regression`, `random_forest`, `gradient_boosting`, `all`, `demo` |
| `--test-size` | `0.25` | hold-out share |
| `--no-balance` | off | disable `class_weight="balanced"` |

### `scripts/retrain_all.py`

| Flag | Default | Meaning |
| --- | --- | --- |
| `--only` | *(all datasets)* | retrain one dataset (stem, e.g. `adult`) |
| `--models` | *(all three)* | restrict model types |
| `--no-balance` | off | disable balanced class weights |

Rebuilds every `.pkl` in `models/` with the current code — hold-out split,
one-hot categoricals, balanced weights and the leakage guard. Run it after a
git pull that touched `fairguard/train.py`, so no artifact is left with
outdated semantics.

### `scripts/visualize_data.py`

| Flag | Default | Meaning |
| --- | --- | --- |
| `--only` | *(all datasets)* | chart one dataset (stem, e.g. `adult`) |

Writes 99 PNG charts into `visualize/` — per dataset: label balance, numeric
distributions, categorical counts, label correlations, correlation heatmap,
missing values, positive-label rate per sensitive group, box plots and a
scatter; plus a PSI drift chart (demo pair) and a text-judge chart.
`visualize/README.md` explains how to read each chart.

### `scripts/run_assessment.py`

| Flag | Default | Meaning |
| --- | --- | --- |
| `--data` | `data/demo_loans.csv` | dataset to assess |
| `--model` | `models/logistic_regression.pkl` | model artifact |
| `--sensitive` | `gender` | sensitive attribute column |
| `--label` | `approved` | target column |
| `--reference` | *(none)* | **different** dataset used as the drift baseline |
| `--threshold` | `0.5` | decision cutoff |
| `--texts` | *(none)* | CSV with a `text` column (and optional `reference`) |
| `--full-dataset` | off | score everything instead of the hold-out |
| `--no-bootstrap` | off | skip the fairness confidence intervals (faster) |
| `--no-mlflow` | off | do not log to MLflow |

### `scripts/make_demo_data.py`

No flags — writes `data/demo_loans.csv` and `data/demo_loans_new.csv`.

### `scripts/download_datasets.py`

No flags — fetches the nine public datasets into `data/` (already downloaded
in this repo; re-running is safe and skips what is present).

### `scripts/estimate_accuracy.py`

| Flag | Default | Meaning |
| --- | --- | --- |
| `--target` | `0.99` | accuracy target used to flag a dataset |
| `--tune` | off | reserved for the hyper-parameter sweep |

---

## Reference — the API endpoints

| Method | Path | Body / notes |
| --- | --- | --- |
| `GET` | `/api/datasets` | `{"datasets": ["adult.csv", ...]}` |
| `GET` | `/api/models` | `{"models": ["gradient_boosting.pkl", ...]}` |
| `POST` | `/api/train` | `{dataset, label, sensitive, model_types, test_size}` → `{results: [...]}` |
| `POST` | `/api/assess` | `{dataset, model, label, sensitive, threshold, reference, use_reference, texts, full_dataset}` → `{run_id, assessment, saved, reference_note}` |
| `GET` | `/api/history` | `{"runs": [...]}` from MLflow |
| `POST` | `/api/review` | `{run_id, decision, note}` → writes the decision onto **that** run |
| `GET` | `/api/report/{run_id}/{fmt}` | `fmt` ∈ `json`, `md`, `card` → `{content, filename}` |

`decision` ∈ `pending`, `Approve`, `Approve with monitoring`, `Needs review`, `Block`.

---

## Reference — configuration knobs

Everything tunable lives in **`fairguard/config.py`**.

| Constant | Default | Meaning |
| --- | --- | --- |
| `RANDOM_STATE` | `42` | global seed |
| `TEST_SPLIT` | `0.25` | training hold-out share |
| `FAIRNESS_DPD_LIMIT` | `0.10` | demographic-parity-difference limit |
| `FAIRNESS_EOD_LIMIT` | `0.10` | equalized-odds-difference limit |
| `MIN_GROUP_SIZE` | `30` | below this a group is flagged as too small |
| `FAIRNESS_BOOTSTRAP` | `200` | bootstrap resamples for the CIs |
| `FAIRNESS_MIN_GROUP_ROWS` | `10` | minimum group size before a CI is computed |
| `FAIRNESS_MIN_BOOTSTRAP_ROWS` | `50` | minimum sample before a CI is computed |
| `PSI_MODERATE` / `PSI_MAJOR` | `0.10` / `0.25` | drift levels |
| `KS_ALPHA` | `0.05` | KS significance level |
| `DRIFT_IMPORTANCE_WEIGHT` | `0.7` | how much drift risk follows feature importance |
| `MIN_ACCURACY` / `MIN_F1` | `0.70` / `0.60` | performance floors |
| `CALIBRATION_BINS` | `10` | ECE bins |
| `RISK_WEIGHTS` | fairness .30, privacy .20, drift .20, performance .15, text .15 | dimension weights |
| `RISK_LEVEL_BOUNDS` | 0.25 / 0.50 / 0.75 | Low / Medium / High boundaries |
| `JUDGE_MODELS` / `JUDGE_TASKS` | see file | which HF model and task per check |

Change a value, re-run the assessment — nothing else needs to move.

---

## 20. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `dataset validation failed: ...` | the CSV has no usable binary label, or the sensitive column is missing | check `--label` / `--sensitive`; `python -c "import pandas as pd; print(pd.read_csv('data/x.csv').columns)"` |
| `ValueError: The least populated classes in y have only 1 member` | a class has a single row | FairGuard now falls back to an unstratified split automatically — if you hit this in your own code, filter those rows out |
| `Feature 'gender' contains missing values` | the sensitive column has NaNs | those rows are now dropped and counted in `dropped_missing_groups` |
| Every PSI is `0.0` | the reference is the assessed dataset | pass a genuinely different `--reference`; the report flags self-comparison |
| Drift tab says "not requested" | no reference selected | pick one in the sidebar / pass `--reference` |
| No fairness confidence intervals | groups or sample too small | that is intentional — the groups are listed under `small_groups` instead |
| `Cannot import name 'decision_tree'` from `sklearn.ensemble` | `DecisionTreeClassifier` lives in `sklearn.tree` | use `from sklearn.tree import DecisionTreeClassifier` |
| SHAP takes minutes | the model-agnostic permutation explainer is in use | check `assessment["explainability"]["explainer"]`; tree/linear models are 60×+ faster |
| `No module named 'transformers'` | optional dependency missing | `pip install transformers torch`, or accept the heuristic backend (the report says which ran) |
| `Address already in use` (api / streamlit) | the port is taken | `python api.py` uses 8000; run `streamlit run app.py --server.port 8502` |
| Reports overwrite each other | two runs in the same second | fixed — a `_2`, `_3` … suffix is added automatically |
| The review decision landed on the wrong run | `run_id` not supplied | fixed — `run_id` is now required and honoured |

---

## 21. Command cheat sheet

```powershell
# ── setup ────────────────────────────────────────────────────────────────
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# ── demo, end to end ─────────────────────────────────────────────────────
python scripts/make_demo_data.py
python scripts/train_demo_model.py
python scripts/run_assessment.py --sensitive gender
python scripts/run_assessment.py --sensitive gender --reference data/demo_loans_new.csv --model models/gradient_boosting.pkl

# ── real data ────────────────────────────────────────────────────────────
python scripts/download_datasets.py
python scripts/train_demo_model.py --data data/adult.csv --label income_gt_50k --sensitive sex
python scripts/run_assessment.py --data data/adult.csv --label income_gt_50k --sensitive sex --model models/adult_logistic_regression.pkl

# ── UIs ──────────────────────────────────────────────────────────────────
python api.py                                    # REST API  :8000
streamlit run app.py                             # dashboard  :8501
cd frontend ; npm install ; npm run dev         # React      :5173

# ── quality ──────────────────────────────────────────────────────────────
python -m pytest tests -q
python scripts/visualize_data.py
python -c "import sys; sys.path.insert(0,'.'); from fairguard import tracking; print(tracking.history().to_string())"
```

---

*FairGuard AI measures a model on a dataset. It does not measure people.*
