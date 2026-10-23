# FairGuard AI — Full Project Explanation

> **FairGuard AI** is a responsible-AI *assessment workbench*: it takes a trained
> classifier and a dataset, and produces one auditable record describing how the
> model behaves — how accurate it is, how fairly it treats groups, why it makes
> the decisions it makes, whether personal data is leaking into it, whether the
> input distribution has moved, and whether its generated text is safe.

Everything runs locally on your machine. There are no paid APIs, no accounts and
no data leaving the machine.

| | |
| --- | --- |
| **Package** | `fairguard/` (the engine) |
| **Front doors** | `scripts/*.py` (CLI) · `api.py` (FastAPI) · `app.py` (Streamlit) · `frontend/` (React) |
| **Audit trail** | `reports/assessment_*.json` + `*.md` + `model_card_*.md` · `mlruns/mlflow.db` |
| **Docs** | [USER_MANUAL.md](USER_MANUAL.md) (how to run) · [ROOT_CAUSES.md](ROOT_CAUSES.md) (the audit) · [README.md](README.md) |

---

## Table of contents

1. [The problem](#1-the-problem)
2. [The idea](#2-the-idea)
3. [Architecture](#3-architecture)
4. [The data flow, step by step](#4-the-data-flow-step-by-step)
5. [Module reference](#5-module-reference)
6. [The risk engine in detail](#6-the-risk-engine-in-detail)
7. [The audit trail](#7-the-audit-trail)
8. [The four front doors](#8-the-four-front-doors)
9. [The twenty-four root causes and their fixes](#9-the-twenty-four-root-causes-and-their-fixes)
10. [The ML decisions that matter](#10-the-ml-decisions-that-matter)
11. [The assessment record](#11-the-assessment-record)
12. [Limits and honest caveats](#12-limits-and-honest-caveats)
13. [Extending FairGuard](#13-extending-fairguard)

---

## 1. The problem

A model with 99 % accuracy can still be unusable, and the usual accuracy
dashboard will happily tell you it is fine.

* It may approve applicants at very different rates depending on gender
  (**fairness**).
* It may rely on a proxy feature nobody noticed (**explainability**).
* It may have been trained on columns containing phone numbers and email
  addresses (**privacy**).
* Its input distribution may have shifted since training, so last quarter's
  accuracy no longer holds (**drift**).
* It may emit toxic or PII-laden text (**LLM safety**).
* And every one of those numbers may have been computed on the very rows the
  model was fitted on, which makes them optimistic and meaningless.

FairGuard exists to make all of that explicit, in one place, with numbers you
can defend in a review meeting.

---

## 2. The idea

> **One call → one record.**

`fairguard.pipeline.run_assessment(...)` takes a model and a dataset and
returns a single dictionary — the *assessment record* — containing:

* the **measurements** of every dimension,
* a **risk score** (0–1) and a **risk level** (Low / Medium / High / Critical),
* a **suggested decision** (Approve / Approve with monitoring / Needs review / Block),
* **plain-language recommendations** describing what to actually do,
* and the **context** needed to interpret the numbers: which rows were scored,
  which reference dataset was used for drift, how big the bootstrap was.

That record is then rendered three ways (JSON, Markdown, model card) and logged
to MLflow. Every consumer — CLI, REST API, Streamlit, React — shows the *same*
record, so all four surfaces agree.

The design principle is **honest measurement over impressive output**: if a
number cannot be computed responsibly (too few rows for a confidence interval,
no drift reference, a heuristic instead of a real model), FairGuard says so
instead of quietly producing a plausible-looking number.

---

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  FRONT DOORS                                                        │
│                                                                      │
│  scripts/*.py        api.py (FastAPI)     app.py (Streamlit)        │
│  (CLI)               (REST)                (dashboard)              │
│                              ▲                    ▲                   │
│                              └────────┬───────────┘                   │
│                                       │                               │
│                              frontend/ (React + Vite + Tailwind)    │
└───────────────────────────────────────┼──────────────────────────────┘
                                        │
┌───────────────────────────────────────▼──────────────────────────────┐
│  ORCHESTRATION                                                        │
│                                                                      │
│  fairguard/pipeline.py   run_assessment() — the single entry point    │
│  fairguard/risk.py       combine → score → level → decision → advice  │
└───────────────────────────────────────┬──────────────────────────────┘
                                        │
        ┌───────────────┬───────────────┼───────────────┬───────────────┐
        ▼               ▼               ▼               ▼               ▼
┌───────────────┐┌──────────────┐┌──────────────┐┌──────────────┐┌─────────────┐
│ performance   ││  fairness    ││explainability││   privacy    ││    drift    │
│ acc/P/R/F1   ││ DPD/EOD + CI ││  SHAP        ││  PII scan    ││  PSI + KS   │
│ ROC-AUC, ECE ││ per-group    ││  global/local││  type-aware  ││ importance- │
│ Brier        ││ small groups ││  exact kinds ││  noisy-OR    ││  weighted   │
└───────────────┘└──────────────┘└──────────────┘└──────────────┘└─────────────┘
        │               │               │               │               │
        └───────────────┴───────────────┴───────────────┴───────────────┘
                                        │
                        ┌───────────────┼────────────────┐
                        ▼               ▼                ▼
                  ┌───────────┐   ┌─────────────┐  ┌───────────┐
                  │  report   │   │  tracking   │  │ judges /  │
                  │ JSON/MD/  │   │  (MLflow)   │  │ llm_eval  │
                  │ model card│   └─────────────┘  └───────────┘
                  └───────────┘
```

**Three layers, one direction of dependency:**

| Layer | Modules | Rule |
| --- | --- | --- |
| **Engine** | `config`, `data`, `train`, `performance`, `fairness`, `explainability`, `privacy`, `drift`, `judges`, `llm_eval`, `risk` | Pure functions on DataFrames. No I/O, no globals, no UI. Each is independently testable. |
| **Orchestration** | `pipeline`, `report`, `tracking` | Wires the engine together, persists and logs the result. |
| **Front doors** | `scripts/`, `api.py`, `app.py`, `frontend/` | Parse input, call `pipeline.run_assessment`, render the record. |

The engine never imports from the orchestration layer, and the orchestration
layer never imports from a front door. That is what makes the engine testable
without a server, a browser or a dataset on disk.

---

## 4. The data flow, step by step

This is what actually happens inside `pipeline.run_assessment()`:

### Step 0 — Preparation

```
df (raw CSV frame)
  │
  ├─ data.add_age_group(df)        → adds an `age_group` bucket column
  └─ data.encode_label(df, label)  → maps the label to 0/1
                                     ({"yes":0,"no":1}, {"low":0,"high":1}, …)
                                     and records the mapping in the assessment
```

Labels are no longer required to be `0`/`1`. A dataset labelled `approved` /
`rejected` or `yes` / `no` is encoded automatically, and the mapping is written
into the record so a reader can decode it.

### Step 1 — Validation

```
data.validate_dataset(df, [sensitive], label, model_features) → list[str]
```

Returns a list of problems (empty = fine). The pipeline raises
`ValueError` with all problems joined, instead of failing later with a
cryptic sklearn error.

### Step 2 — Encoding

```
train.encode_features(df, features=raw_features, columns=features) → X
```

Numeric columns pass through. **Categorical columns are one-hot encoded** if
their cardinality is ≤ 20 (`MAX_CATEGORICAL_CARDINALITY`). A `ColumnTransformer`
inside the pipeline handles the scaling/encoding at fit time, so the same
transform is applied to the hold-out.

### Step 3 — Scoring the model

```
y_prob = model.predict_proba(X_eval)[:, 1]
y_pred = (y_prob >= threshold).astype(int)
```

`X_eval` is the **hold-out** when one is supplied, otherwise the whole frame.
Which one was used is recorded in `assessment["evaluation"]`.

### Step 4 — The six measurements

| Dimension | Function | Produces |
| --- | --- | --- |
| Performance | `performance.evaluate_performance` | accuracy, precision, recall, F1, ROC-AUC, Brier, ECE, confusion matrix |
| Fairness | `fairness.assess_fairness` | DPD, EOD, per-group rates, bootstrap CIs, small-group flags |
| Explainability | `explainability.global_importance_with_meta` | mean \|SHAP\| per feature + which explainer ran |
| Privacy | `privacy.scan_pii` | PII-like columns with severity and masked samples |
| Drift | `drift.assess_drift` | PSI + level and KS p-value per feature (only when a reference exists) |
| Text | `llm_eval.evaluate_texts` | per-text scores for toxicity / bias / hate speech / PII |

### Step 5 — The risk engine

```
risks = {"performance": …, "fairness": …, "privacy": …}
       + "drift" if a reference was supplied
       + "text"   if texts were supplied

assessment["risk"] = risk.compute_risk(risks)
assessment["recommendations"] = risk.build_recommendations(assessment)
assessment["decision"] = risk.deployment_decision(assessment["risk"]["level"])
```

### Step 6 — Persistence and logging

```
report.save_report(assessment)     → reports/assessment_<stamp>.{json,md} + model_card_<stamp>.md
tracking.log_assessment(assessment) → an MLflow run in mlruns/mlflow.db
```

---

## 5. Module reference

### `fairguard/config.py` — one place for every knob

Paths (`DATA_DIR`, `MODELS_DIR`, `REPORTS_DIR`, `MLRUNS_DIR`), the seed
(`RANDOM_STATE`), the fairness limits (`FAIRNESS_DPD_LIMIT`,
`FAIRNESS_EOD_LIMIT`, `MIN_GROUP_SIZE`, `FAIRNESS_BOOTSTRAP`), the drift
thresholds (`PSI_MODERATE`, `PSI_MAJOR`, `KS_ALPHA`), the performance floors
(`MIN_ACCURACY`, `MIN_F1`), the risk weights (`RISK_WEIGHTS`,
`RISK_LEVEL_BOUNDS`) and the judge models (`JUDGE_MODELS`, `JUDGE_TASKS`).

*Why it exists:* "everything a review meeting might want to tune lives here, in
one place". No magic numbers are scattered through the engine.

### `fairguard/data.py` — dataset intake

| Function | Job |
| --- | --- |
| `load_csv(path)` | Read a CSV, coerce, keep it simple. |
| `is_binary_label(values)` | True when the label has ≤ 2 distinct values. |
| `binary_label_mapping(values)` | `{original: 0/1}` for a 2-valued label. |
| `encode_label(df, label)` | Apply the mapping; returns `(df, mapping)`. |
| `add_age_group(df, column="age")` | Bucket `age` into `age_group` for group metrics. |
| `validate_dataset(...)` | Returns a list of human-readable problems. |
| `profile_dataset(df)` | Row/column counts, dtypes, missing values, numeric summary. |

### `fairguard/train.py` — feature selection, training, artifacts

| Function | Job |
| --- | --- |
| `make_model(model_type, balance_classes=True)` | Build the sklearn `Pipeline` (scaler/encoder + estimator). `class_weight="balanced"` by default. |
| `encode_features(df, columns, features, categorical)` | One-hot encode low-cardinality categoricals. |
| `pick_features(df, label, sensitive, exclude, include_categorical=True)` | Choose the model's inputs: numeric **and** categorical, minus the label, minus the sensitive column, minus obvious identifiers. |
| `split_data(X, y, test_size, random_state)` | Stratified split that degrades gracefully to unstratified when a class has < 2 members. Returns `(X_train, X_test, y_train, y_test, can_stratify)`. |
| `train_model(...)` | Fit and return `(model, X_train, X_test, y_train, y_test)`. |
| `make_holdout(df, ...)` | Recompute *the same* split without training, and store the held-out **positions**. |
| `holdout_rows(df, holdout)` | Slice those positions out of a frame. |
| `save_model(...)` / `load_model(path)` | Persist a dict artifact: model, features, raw_features, model_type, holdout, meta. |

The artifact is a plain dict (pickle) — inspectable, no framework lock-in.

### `fairguard/performance.py` — accuracy *and* calibration

```
evaluate_performance(y_true, y_pred, y_prob) →
    accuracy, precision, recall, f1, roc_auc, confusion_matrix,
    brier_score, calibration_error
```

**Expected Calibration Error (ECE)** is the interesting one. It splits the
predicted probabilities into `CALIBRATION_BINS` bins and, in each bin, compares
the *empirical positive rate* with the *mean predicted probability*. A model
that says "70 % chance" for every applicant but is right only half the time
scores 0.20 ECE — and its threshold-based decisions are therefore suspect.

`performance_risk(metrics)` = `max(accuracy_gap, f1_gap)` against the floors.

### `fairguard/fairness.py` — disparity, with uncertainty

**Demographic Parity Difference (DPD)** — the gap in selection rates between
the best- and worst-served group. "Are groups approved at the same rate?"

**Equalized Odds Difference (EOD)** — the gap in true-positive rates (or
false-positive rates). "Do groups get the same *quality* of outcome?"

Both are computed with `fairlearn.MetricFrame`, which gives per-group
selection rate, accuracy, precision and recall in one pass.

Then the part that makes the numbers *usable*:

* **Bootstrap confidence intervals.** `bootstrap_intervals()` resamples within
  each group, `FAIRNESS_BOOTSTRAP` times, and reports the 2.5 % / 97.5 %
  quantiles of the disparity. A DPD of 0.08 with a CI of `[0.01, 0.19]` is a
  *different* finding from a DPD of 0.08 with a CI of `[0.07, 0.09]`.
* **Gating.** Intervals are only computed when every group has at least
  `FAIRNESS_MIN_GROUP_ROWS` (10) rows and the sample has at least
  `FAIRNESS_MIN_BOOTSTRAP_ROWS` (50). Below that the interval is noise, so
  the groups are reported as `small_groups` instead.
* **Straddle rule.** `_effective_disparity(point, ci, limit)` decides what the
  risk engine should believe:
  * CI entirely below the limit → the point estimate is trustworthy → use it.
  * CI entirely above the limit → the violation is real → use it.
  * CI straddles the limit → the honest answer is "we cannot tell" → use the
    CI midpoint (which is ≤ the limit only if the evidence is genuinely
    ambiguous).
* **Missing values.** Rows whose sensitive attribute is missing are dropped
  and *counted* (`dropped_missing_groups`) instead of crashing the run.

### `fairguard/explainability.py` — SHAP, at the right speed

FairGuard models are sklearn `Pipeline`s (encoder → scaler → estimator). The
old code looked for `isinstance(model, LogisticRegression)`, which never
matches a pipeline — so **every** model silently fell back to SHAP's
model-agnostic `PermutationExplainer`. Measured on `communities.csv` (99
features, 1994 rows): **234.5 s** versus **3.5 s** for `TreeExplainer` — a
67× difference that made the tool unusable on real datasets.

The fix is to *unwrap* the pipeline:

```
split_pipeline(model) → (estimator, transform)
```

then choose the explainer by the **estimator's** type:

| Estimator | Explainer | Why |
| --- | --- | --- |
| `DecisionTreeClassifier`, `RandomForestClassifier`, `ExtraTreesClassifier`, `GradientBoostingClassifier`, `HistGradientBoostingClassifier` | `shap.TreeExplainer` | Exact, and 60–100× faster. |
| `LogisticRegression`, `LinearRegression` | `shap.LinearExplainer` | Exact for linear models. |
| anything else | `shap.Explainer(model.predict_proba, …)` | Permutation fallback — correct, just slow. |

`Explainer` is a thin adapter that applies the pipeline's transform before
handing the matrix to SHAP, so the feature names stay meaningful.

`global_importance_with_meta(model, X)` returns the importance **and** the
metadata a reader needs: which explainer ran (`explainer`), how many
background rows it used (`n_background`), how many rows were explained
(`n_explained`).

`local_explanation(model, X, row_index)` gives the signed per-feature
contributions for one row — the "why did *this* person get rejected?" view.
The explained row is removed from its own background set, so it is not
explained against itself.

### `fairguard/privacy.py` — type-aware PII detection

| Pattern | Severity |
| --- | --- |
| `credit_card` | 1.0 |
| `aadhaar` | 1.0 |
| `email` | 0.9 |
| `phone` | 0.8 |
| `ip_address` | 0.4 |

**The type-aware rule.** A 16-digit `account_id` is *not* a credit card. The
old scanner matched the digit pattern and reported severity 1.0 — a false
alarm on an identifier column. Now:

* Text columns are scanned with **all** patterns.
* Numeric columns are scanned with the **strong** patterns (email, credit
  card) only.
* Numeric columns are scanned with the **ambiguous** patterns (phone, aadhaar,
  IP) **only if the column name justifies it** — `PII_NAME_TRIGGERS` maps a
  type to name fragments (`card`, `phone`, `aadhaar`, `ip`, …).
* A column whose *name* looks sensitive is reported as
  `sensitive_column_name` even when no pattern matched.

`privacy_risk` combines the findings with a **noisy-OR**
(`1 - Π(1 - severity)`) so many small leaks do not saturate the score, and
samples are masked before they leave the function.

### `fairguard/drift.py` — PSI and KS, weighted by what matters

```
psi(reference, current, bins=10)              → 0..1
categorical_psi(reference, current)           → 0..1
ks(reference, current)                        → (statistic, p_value)
psi_level(value)                              → "stable" | "moderate" | "major"
assess_drift(reference, current, features)     → {feature: {psi, psi_level, ks_pvalue}}
```

**PSI (Population Stability Index)** bins the reference distribution and
measures how much mass moves between bins. Thresholds: `< 0.10` stable,
`0.10–0.25` moderate, `> 0.25` major.

**KS (Kolmogorov–Smirnov)** compares the two distributions directly and gives
a p-value — is the difference statistically distinguishable from noise?

**The aggregation problem.** The old risk function took the *mean* severity
over all features. One feature with a major shift among twenty gave a risk of
0.05 — critical drift diluted into invisibility. Now:

```
drift_risk = w · aggregate_severity + (1 - w) · peak_severity
w = DRIFT_IMPORTANCE_WEIGHT (0.7) when feature importance is available
```

Drift in a feature the model actually uses matters more than drift in a feature
it ignores. `aggregate_importance` folds one-hot SHAP importances back onto
the raw columns (using `raw_features`) so "income_lakhs_female" and
"income_lakhs_male" are judged as one feature, not two.

### `fairguard/judges.py` and `fairguard/llm_eval.py` — text safety

Five checks: **toxicity**, **bias**, **hate speech**, **PII**, and
**hallucination** (needs a reference text).

Two backends per check:

* `HeuristicJudge` — an offline word/phrase matcher. Fast, deterministic,
  no downloads.
* `TransformerJudge` — a Hugging Face model, used when `transformers` is
  installed.

**The task-type bug.** Every judge used to call
`pipeline("text-classification", ...)`. But the PII model
(`iiiorg/piiranha-v1-detect-personal-information`) is a **token-classification**
model and the hallucination model (`vectara/hallucination_evaluation_model`)
is a **pair** model needing a second text. Calling them as text-classification
raised at runtime. `config.JUDGE_TASKS` now declares the correct task per
check and the judge dispatches on it.

**The length bug.** The heuristic toxicity score was `hits / words`. The word
"stupid" alone scored 1.0; the same word in a 41-word sentence scored 0.12 —
the score depended on the length of the text, not its toxicity. The new
`_word_risk` takes `max(density, count)` where the density term saturates at
`DENSITY_SATURATION` (0.2) and the count term is `1 - e^(-hits)`.

**The silent fallback.** `evaluate_texts` used to report the backend of the
*first* check only, so a PII judge that fell back to heuristics was invisible.
It now returns `backends` per check plus a `backend` summary.

### `fairguard/risk.py` — one score, honest coverage

```
compute_risk(risks) →
    score, level, contributions, dimensions_assessed, coverage, weights_used
```

```
score = Σ (risk_d × weight_d) / Σ weight_d      over the dimensions assessed
coverage = Σ weight_d / Σ all weights
```

The **coverage** field is the honesty mechanism. Because the score is
normalised over the dimensions that were *actually* assessed, the same
underlying fairness risk produced 0.18 "Low" with five dimensions and 0.2769
"Medium" with three. Now the number carries its own caveat: two assessments
are only comparable when their coverage matches.

**Levels and decisions:**

| Score | Level | Suggested decision |
| --- | --- | --- |
| ≤ 0.25 | Low | Approve |
| ≤ 0.50 | Medium | Approve with monitoring |
| ≤ 0.75 | High | Needs review |
| > 0.75 | Critical | Block |

`build_recommendations(assessment)` turns the findings into sentences a human
can act on — fairness violations (both DPD **and** EOD), small groups,
performance floors, poor calibration, PII columns, major/moderate drift,
flagged texts.

### `fairguard/pipeline.py` — the single entry point

```python
run_assessment(
    df, model, features,
    label="approved", sensitive="gender",
    reference=None,               # drift baseline; None = skip drift
    model_id="model", model_version="1.0", dataset_id="dataset",
    threshold=0.5,
    include_explainability=True, run_mlflow=True, extra_artifacts=None,
    text=None,                    # pre-computed text assessment
    raw_features=None,            # pre-encoding columns
    eval_df=None,                 # the hold-out to score on
    n_boot=None,                  # bootstrap resamples
) → dict
```

`compare_models(models, df, **kwargs)` runs the same assessment for several
models on one dataset — useful for a model bake-off.

### `fairguard/report.py` — three renderings of one record

* `to_json` — the raw record (machine-readable, the source of truth).
* `build_markdown` — the audit-ready report, with an **Evaluation scope**
  section, coverage, fairness CIs, calibration numbers, the explainer kind,
  and a warning when the drift reference is the assessed dataset.
* `build_model_card` — the compact, shareable summary in the spirit of
  *Model Cards for Model Reporting* (Mitchell et al., 2019).

`save_report` writes all three, adding a `_2`, `_3` … suffix when two
assessments land in the same second (they used to overwrite each other).

### `fairguard/tracking.py` — MLflow without a server

MLflow points at a **SQLite** database in `mlruns/`, so no tracking server is
needed. Each assessment becomes a run with params (model, version, dataset,
sensitive attribute), tags (risk level, decision, reviewer decision, **evaluation
scope**, **explainer**) and metrics (risk score, every performance metric, the
fairness point estimates **and** CI bounds, small-group count, max PSI, major
drift count, risk coverage).

`history()` returns a tidy DataFrame for the dashboard.

---

## 6. The risk engine in detail

```
                        ┌─────────────────────────────────────────┐
                        │            risk.compute_risk            │
   performance  ───────▶│                                         │
   fairness     ───────▶│  score = Σ (r_d × w_d) / Σ w_d          │──▶ score
   privacy      ───────▶│  (over the dimensions actually assessed)│    level
   drift        ───────▶│                                         │    contributions
   text         ───────▶└─────────────────────────────────────────┘    coverage
                        └─────────────────────────────────────────┘
                                      │
                                      ▼
                        ┌─────────────────────────────────────────┐
                        │        risk.build_recommendations        │
                        │  "Fairness: … exceed the configured      │──▶ list[str]
                        │   limits — consider Fairlearn's          │
                        │   ThresholdOptimizer."                   │
                        └─────────────────────────────────────────┘
```

**Default weights** (`config.RISK_WEIGHTS`):

| Dimension | Weight | Rationale |
| --- | --- | --- |
| Fairness | 0.30 | The dimension the whole project exists for. |
| Privacy | 0.20 | A hard failure regardless of how good the model is. |
| Drift | 0.20 | Invalidates every other measurement if the data moved. |
| Performance | 0.15 | Necessary but not sufficient. |
| Text | 0.15 | Only present in LLM pipelines. |

**Why normalise at all?** Because the dimensions are not all present in every
run — a tabular loan model has no text dimension, so its maximum achievable
score is lower. Normalising over what was assessed gives a 0–1 score. The
`coverage` field then says "this 0.62 covers 65 % of the weight", so nobody
reads a 3-dimension score as equal to a 5-dimension score.

---

## 7. The audit trail

FairGuard is designed to be *re-checkable* by a reviewer who was not present.

| Artefact | Where | What it holds |
| --- | --- | --- |
| Assessment JSON | `reports/assessment_<stamp>.json` | The complete record — every number, every context field. |
| Assessment Markdown | `reports/assessment_<stamp>.md` | The human-readable report. |
| Model card | `reports/model_card_<stamp>.md` | The shareable summary. |
| MLflow run | `mlruns/mlflow.db` | Params, tags and metrics for every assessment, queryable over time. |
| Model artifact | `models/*.pkl` | The model, its features, its hold-out split and its training metadata. |

**The reviewer loop** (Streamlit → API):

1. An analyst runs an assessment → a record + a report are written.
2. A reviewer opens the *Results* / *Report* tab and reads the numbers.
3. They choose a decision (Approve / Approve with monitoring / Needs review /
   Block) and optionally a note.
4. The decision is written onto **that** assessment record (`/api/review` with
   the `run_id`) and re-logged to MLflow.
5. The exported report carries the decision and the note.

---

## 8. The four front doors

### CLI (`scripts/`)

| Script | Job |
| --- | --- |
| `make_demo_data.py` | Write the synthetic demo dataset and a drifted copy. |
| `download_datasets.py` | Fetch the nine public UCI / ProPublica datasets into `data/`. |
| `train_demo_model.py` | Train one or more models and persist the artifacts (with the hold-out). |
| `run_assessment.py` | Run a full assessment and print the risk, scope, coverage and recommendations. |
| `estimate_accuracy.py` | Cross-validated accuracy estimate per dataset/model. |

### REST API (`api.py`)

FastAPI + CORS, version `0.3.0`.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/datasets` | List the CSVs in `data/`. |
| `GET` | `/api/models` | List the trained artifacts. |
| `POST` | `/api/train` | Train and persist models; returns the hold-out size. |
| `POST` | `/api/assess` | Run an assessment; returns `{run_id, assessment, saved, reference_note}`. |
| `GET` | `/api/history` | The MLflow history. |
| `POST` | `/api/review` | Attach a reviewer decision to a **specific** `run_id`. |
| `GET` | `/api/report/{run_id}/{fmt}` | Export `json` / `md` / `card`. |

### Streamlit (`app.py`)

Ten tabs (Overview, Fairness, Explainability, Privacy, Drift, Performance,
Text & LLM, Review, History, Report) with Plotly charts and live downloads.
It also auto-switches to the dataset a model was trained on when the selected
dataset does not contain the model's features.

### React (`frontend/`)

Vite + React + Tailwind with five pages — Dashboard, Train, Assess, Results,
History — talking to the FastAPI backend through `src/api.ts`.

### Windows helper scripts

| Script | Job |
| --- | --- |
| `start.bat` | The one-click demo: generates `data/demo_loans.csv` and the three demo models if they are missing, then starts the FastAPI backend (8000), the Streamlit dashboard (8501) and — when `frontend/node_modules` exists — the React frontend (5173). |
| `stop.bat` | Kills all three servers by port. |

### Windows helper scripts

`start.bat` brings up the whole demo in one go: it generates the demo data and
models when they are missing, then starts the FastAPI backend (port 8000), the
Streamlit dashboard (port 8501) and, when `frontend/node_modules` exists, the
React frontend (port 5173). `stop.bat` kills all three by port.

---

## 9. The twenty-four root causes and their fixes

The full write-up with measured evidence is in **[ROOT_CAUSES.md](ROOT_CAUSES.md)**.
The summary:

| # | Severity | Root cause | Fix |
| --- | --- | --- | --- |
| RC-01 | **Critical** | SHAP always used the permutation explainer (pipelines never matched the `isinstance` check) — 234.5 s vs 3.5 s | Unwrap the pipeline, choose `TreeExplainer` / `LinearExplainer` by estimator type; report the explainer kind |
| RC-02 | **Critical** | Assessments scored the model on the rows it was trained on | The artifact carries its hold-out; assessments score it by default and state the scope |
| RC-03 | **High** | `pick_features` dropped every categorical column | One-hot encode low-cardinality categoricals and keep them |
| RC-04 | **High** | Recommendations only checked DPD — EOD 0.30 was reported as "no blocking issues" | Check both disparities, against both limits |
| RC-05 | **High** | Risk score normalised over the assessed dimensions, so identical findings scored 0.18 vs 0.2769 | Report `coverage` and `dimensions_assessed` alongside the score |
| RC-06 | **High** | `drift_risk` averaged severity, diluting one critical feature to 0.05 | Blend the aggregate with the peak, weighted by feature importance |
| RC-07 | **High** | Drift compared the dataset with itself → every PSI exactly 0 | The reference must be a different dataset; self-comparison is flagged as a baseline |
| RC-08 | **Medium** | Privacy scanner flagged a 16-digit `account_id` as a credit card | Type-aware scanning + column-name triggers |
| RC-09 | **Medium** | `validate_dataset` crashed on mixed-type labels | `is_binary_label` on the distinct values; string labels accepted and encoded |
| RC-10 | **Medium** | `train_test_split` aborted when a class had < 2 members | Degrade to an unstratified split |
| RC-11 | **Medium** | Fairness crashed on a NaN sensitive column | Drop and count the affected rows |
| RC-12 | **Medium** | The transformer judge called every model a text-classification model | Per-kind task dispatch (`config.JUDGE_TASKS`) |
| RC-13 | **Medium** | The PII judge was dead code | `pii` added to the default kinds |
| RC-14 | **Medium** | Only the first judge's backend was reported | Per-kind `backends` in the result |
| RC-15 | **Medium** | Heuristic toxicity depended on text length | `max(density, count)` scoring |
| RC-16 | **Medium** | `POST /api/review` ignored `run_id` and overwrote the newest report | Look the run up by id (exact → suffixed → substring) |
| RC-17 | **Medium** | No probability calibration check | ECE + Brier score, plus a recommendation when ECE > 0.15 |
| RC-18 | **Medium** | Fairness numbers had no uncertainty | Stratified bootstrap CIs, gated on sample size |
| RC-19 | **Low** | No class-imbalance handling | `class_weight="balanced"` by default |
| RC-20 | **Low** | The dashboard profiled the wrong dataset after the auto-switch | The assessment returns the frame it actually used |
| RC-21 | **Medium** | `expected_calibration_error` compared a bin's *accuracy* with its mean confidence, so a perfectly calibrated model scored 0.5 instead of 0.0 | Compare the bin's **empirical positive rate** with its mean confidence |
| RC-22 | **Medium** | The drift self-reference check compared against the re-encoded frame, so it never fired | Capture the original frame identity before re-encoding and compare against that |
| RC-23 | **Critical** | `adult.csv` ships the raw label `income` beside `income_gt_50k`; keeping categoricals made it a feature, so hold-out accuracy was a bogus 1.000 | `label_duplicate_columns` guard: a bijective rename of the label is dropped, even from an explicit feature list; **the pipeline repeats the check at assessment time** (stale artifacts get `assessment["leakage"]`, a leading "Leakage" recommendation and ⚠️ sections in the report and model card); `scripts/retrain_all.py` rebuilt every artifact with the current code |
| RC-24 | **High** | `fairness._per_group_rates` iterated `for group in groups:` — every row — making the bootstrap O(rows²): 639 s of a 645 s Adult assessment | Iterate `np.unique(groups)` instead of the rows; 12 000 rows bootstrap in 0.86 s, the whole Adult run in 9.4 s |

RC-21 and RC-22 were found *by the new tests* while fixing the first twenty.
RC-23 was found while re-measuring the accuracy table for the README — a
1.000 on census data was too good to be true, and it was. RC-24 was found by
profiling that same re-run: 99 % of the runtime sat in one loop that walked
the rows instead of the distinct groups.

---

## 10. The ML decisions that matter

### Exact explainers over generic ones

SHAP has three families: model-specific (tree / linear), and model-agnostic
(permutation). The agnostic one works for anything and is *correct*, but it
re-evaluates the model thousands of times. Because FairGuard's models are
pipelines, the old type check never matched and everything fell back to it.
Unwrapping the pipeline and switching to the exact explainer is the single
largest performance win in the project — and it also makes the explainer
*exact* rather than approximate.

### Out-of-sample by default

The hold-out is stored **in the artifact**, not recomputed at assessment time.
That matters: recomputing the split later would silently break if
`RANDOM_STATE`, `TEST_SPLIT` or the feature set changed, and the assessment
would quietly score the wrong rows. Storing the *positions* (not the index
labels) also survives the CSV being re-read.

`make_holdout` deliberately mirrors `train_model`'s split logic, and a test
asserts the two agree — that is what keeps the stored hold-out honest.

### Intervals, not just points

A point estimate without an interval invites over-reading small differences.
Bootstrapping *within* groups (rather than over the whole sample) respects the
group structure: the resampled distribution of a disparity should reflect the
sampling variability of each group separately.

The gating matters just as much: an interval computed from 8 rows is
arithmetic, not information. Below the thresholds FairGuard reports the groups
as small instead.

### Importance-weighted drift

Drift in a feature the model uses is a different event from drift in a
feature it ignores. The `DRIFT_IMPORTANCE_WEIGHT` blend (0.7 aggregate /
0.3 peak) captures both: the peak guards against one critical feature being
averaged away, the aggregate guards against one noisy feature dominating.

Folding one-hot importances back onto raw columns is essential here — without
it, a categorical feature with 30 levels looks like 30 separate features and
the aggregation is meaningless.

### Balanced class weights

`class_weight="balanced"` compensates for the class imbalance that makes
accuracy misleading (a model that always predicts "reject" scores 80 %
accuracy on an 80/20 dataset). It can be turned off with `--no-balance` to
compare.

### Encoding categoricals instead of discarding them

Dropping categorical features (the old behaviour) meant `adult.csv` lost
workclass, education, marital_status, occupation, relationship, race, sex,
native_country — i.e. most of the signal. One-hot encoding with a cardinality
cap keeps them without exploding the feature space.

---

## 11. The assessment record

```jsonc
{
  "model_id": "gradient_boosting",
  "model_version": "gradient_boosting",
  "dataset_id": "demo_loans",
  "sensitive_attribute": "gender",
  "threshold": 0.5,

  // how the numbers were produced
  "evaluation": {
    "scope": "holdout",          // or "full_dataset"
    "n_rows": 500,               // rows actually scored
    "n_dataset_rows": 2000,      // rows in the dataset
    "in_sample": false
  },

  "performance": {
    "accuracy": 0.992, "precision": 0.99, "recall": 0.98, "f1": 0.985,
    "roc_auc": 0.999, "brier_score": 0.011, "calibration_error": 0.018,
    "confusion_matrix": [[...], [...]]
  },

  "fairness": {
    "demographic_parity_difference": 0.02,
    "equalized_odds_difference": 0.38,
    "demographic_parity_difference_ci": [0.0011, 0.0541],
    "equalized_odds_difference_ci": [0.0023, 0.0825],
    "by_group": { "selection_rate": {...}, "accuracy": {...}, ... },
    "group_sizes": { "male": 269, "female": 231 },
    "small_groups": [],
    "n_boot": 200, "ci_level": 95
    // "dropped_missing_groups": 5   ← present only when rows were dropped
  },

  "explainability": {
    "global_importance": { "income_lakhs": 0.31, ... },
    "explainer": "tree",
    "n_background": 300,
    "n_explained": 300
  },

  "privacy": {
    "findings": [ { "column": "email", "type": "email", "matches": 2000, "severity": 0.9 } ],
    "risk": 0.42,
    "columns_scanned": 12,
    "pii_columns": ["email", "full_name", "phone"]
  },

  "drift": { "income_lakhs": { "psi": 0.31, "psi_level": "major", "ks_pvalue": 0.002 } },
  "drift_reference": {
    "same_as_assessed": false,
    "n_rows": 2000,
    "note": "Reference dataset differs from the assessed dataset."
  },

  "text": { "backend": "heuristic", "backends": {...}, "risk": 0.4, "flagged": 1, "n_texts": 4, ... },

  "risk": {
    "score": 0.7335, "level": "High",
    "contributions": { "fairness": 0.30, "drift": 0.18, ... },
    "dimensions_assessed": ["drift", "fairness", "performance", "privacy"],
    "coverage": 0.85,
    "weights_used": { ... }
  },

  "recommendations": ["Fairness: ...", "Drift: ..."],
  "decision": "Needs review",
  "reviewer_decision": null,
  "label_mapping": { "approved": 1, "rejected": 0 },
  "assessed_at": "2026-10-22T14:12:37+00:00",
  "mlflow_run_id": "…"
}
```

---

## 12. Limits and honest caveats

FairGuard is an **assessment tool**, not a compliance product.

* **It measures a model on a dataset.** A fairness number here is a statistic
  about predictions, not a conclusion about people.
* **It does not remove bias.** It surfaces it. Mitigation (reweighing, threshold
  optimisation, better data) is a modelling decision a human has to make.
* **The text judges are not safety guarantees.** The heuristic backend is a
  word matcher; the transformer backend is a single model's opinion. Both
  produce false positives and false negatives.
* **Drift needs a reference.** Without one, FairGuard says "drift was not
  assessed" rather than inventing a number.
* **Small samples are weak evidence.** That is why CIs are gated and groups are
  flagged.
* **The risk score is a heuristic aggregation.** The weights are a starting
  point for discussion, not a validated risk model. The `coverage` field tells
  you how much of the total weight was even assessed.
* **No production hardening.** No auth, no rate limiting, no secret
  management, no multi-tenancy. It is a research/teaching workbench.

---

## 13. Extending FairGuard

**Add a dimension.** Write a pure function `my_dimension(df, model) → dict`
with a 0–1 risk, register a weight in `config.RISK_WEIGHTS`, call it from
`pipeline.run_assessment`, add it to the risk dict and to `build_recommendations`.
The report, model card and MLflow logging pick it up automatically (the
performance/fairness loops are generic).

**Add a dataset.** Drop a CSV in `data/` with a binary label and a sensitive
column, then train and assess. No code change.

**Add a model type.** Implement it in `train.make_model`, add it to
`train.MODEL_TYPES`. For a tree or linear estimator the exact SHAP explainer
works automatically — that is the whole point of the estimator-type dispatch.

**Add a text check.** Add the kind to `config.JUDGE_MODELS` **and**
`config.JUDGE_TASKS`, add the vocabulary / logic to `HeuristicJudge`, and add
the kind to `llm_eval.DEFAULT_KINDS`. The per-kind backend reporting means a
new check can never silently hide.

**Tune the thresholds.** Everything is in `fairguard/config.py`. Change a
value, re-run the assessment, and the reports, model cards, MLflow metrics and
recommendations all follow.

---

*FairGuard AI measures a model on a dataset. It does not measure people.*
