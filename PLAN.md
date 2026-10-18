# FairGuard AI — Implementation Plan

Plan of record for building the workbench. Each step becomes **one pull request**
into `main` — code + tests + a detailed PR description — and steps are followed
in order. The acceptance check of a step must pass before its PR is merged.

## Definition of done (per step)

- [ ] Code implements one small, human-explainable piece of the workbench
- [ ] Tests added and passing locally (`python -m pytest -q`)
- [ ] PR opened with summary / changes / test evidence
- [ ] PR merged into `main` after checks pass

## Steps

| # | Branch | Deliverable | Acceptance check |
|---|--------|-------------|------------------|
| 01 | `step-01-scaffold` | Repo skeleton: `fairguard/` package, `config.py`, README, PLAN, requirements | `python -c "import fairguard"` |
| 02 | `step-02-data` | `data.py`: load, validate, profile, age groups + `scripts/make_demo_data.py` | `pytest tests/test_data.py -q` |
| 03 | `step-03-model` | `train.py`: train / save / load **two** baseline models (logistic regression + random forest) | `pytest tests/test_train.py -q` |
| 04 | `step-04-performance` | `performance.py`: accuracy, precision, recall, F1, ROC-AUC, confusion matrix | `pytest tests/test_performance.py -q` |
| 05 | `step-05-fairness` | `fairness.py`: Fairlearn group-wise metrics, disparity measures, group performance | `pytest tests/test_fairness.py -q` |
| 06 | `step-06-explainability` | `explainability.py`: SHAP global + local explanations | `pytest tests/test_explainability.py -q` |
| 07 | `step-07-privacy` | `privacy.py`: PII / sensitive-column detection + privacy-risk summary | `pytest tests/test_privacy.py -q` |
| 08 | `step-08-drift` | `drift.py`: PSI + KS drift per feature (optional Evidently report) | `pytest tests/test_drift.py -q` |
| 09 | `step-09-risk` | `risk.py`: weighted risk score, recommendations, deployment decision | `pytest tests/test_risk.py -q` |
| 10 | `step-10-report` | `report.py`: JSON (SRS assessment-record schema), Markdown report, automated model card | `pytest tests/test_report.py -q` |
| 11 | `step-11-tracking` | `tracking.py`: MLflow logging of runs, params, metrics, artifacts | `pytest tests/test_tracking.py -q` |
| 12 | `step-12-pipeline` | `pipeline.py` + `scripts/run_assessment.py`: one-call end-to-end assessment and model comparison | `pytest tests/test_pipeline.py -q` |
| 13 | `step-13-judges` | `judges.py`: text judge registry (toxicity, bias, hate speech, hallucination, PII) with heuristic + optional transformers backends | `pytest tests/test_judges.py -q` |
| 14 | `step-14-llm-eval` | `llm_eval.py`: score texts / model outputs, aggregate text-safety score (optional text generation) | `pytest tests/test_llm_eval.py -q` |
| 15 | `step-15-dashboard` | `app.py` part 1: upload & configure + fairness / explainability / performance tabs | `streamlit run app.py` smoke + `pytest -q` |
| 16 | `step-16-dashboard-full` | `app.py` part 2: privacy, drift, risk & review decision, model comparison, text/LLM tab, MLflow history, report download | manual smoke + `pytest -q` |
| 17 | `step-17-final` | Integration test, GitHub Actions CI, docs polish, README refresh | `pytest -q` all green |

## Fixed design decisions

- **One risk score** = weighted average of per-dimension risk, weights in
  `config.RISK_WEIGHTS`, normalized over the dimensions that were assessed.
  Levels: Low ≤ 0.25, Medium ≤ 0.50, High ≤ 0.75, else Critical.
- **Assessment record** follows the SRS schema: `model_id`, `model_version`,
  `dataset_id`, `sensitive_attribute`, `fairness_metrics` (JSON),
  `feature_importance` (JSON), `privacy_findings` (JSON), `drift_scores` (JSON),
  `performance_metrics` (JSON), `risk_score`, `risk_level`, `recommendations`,
  `assessed_at`.
- **Model comparison** is first-class (MVP requirement): the pipeline assesses a
  list of models and the dashboard shows them side by side.
- **Judges are pluggable**: a `transformer` backend when `transformers` + `torch`
  are installed (model ids in `config.JUDGE_MODELS`), otherwise a clearly-labelled
  heuristic backend, so the workbench always runs offline.
- **Drift** is PSI + KS (SRS). An optional Evidently report can be attached when
  the package is available — useful, but never a hard requirement.
- **No paid APIs, no keys** — everything runs locally.
- **Original university documents are not committed** (they contain personal
  contact details and the repository is public).

## Out of scope (per SRS)

Automatic retraining or bias mitigation in production, legal/regulatory
certification, deep-learning-specific explainability for all architectures.

## Status — 2026-10-18

All 17 steps above are implemented and locally committed on `main` (one commit per
module) and verified:

- `python -m pytest tests -q` → **59 passed** (fully offline, no keys)
- end-to-end: demo data → train all three models → `scripts/run_assessment.py`
  → demo accuracy **0.992** / risk Medium / decision **Approve with monitoring**, with JSON + Markdown + model card exported
- dashboard: Streamlit `AppTest` run + "Run assessment" click → no exceptions, all tabs render
- CI: `.github/workflows/tests.yml` runs the suite on every push / pull request

### Datasets (10 total, one-command download via `scripts/download_datasets.py`)

| Dataset | Rows | Label | Sensitive | Best model | Accuracy |
|---|---|---|---|---|---|
| demo_loans (synthetic) | 2,000 | approved | gender | gradient_boosting | 0.992 |
| student_performance | 1,044 | G3_pass | sex | random_forest | 0.900 |
| bank_marketing | 45,211 | subscribed | age_group | gradient_boosting | 0.888 |
| adult | 48,842 | income_gt_50k | sex | gradient_boosting | 0.846 |
| taiwan_credit | 30,000 | default_next_month | gender | gradient_boosting | 0.820 |
| heart | 303 | disease | sex | random_forest | 0.855 |
| german | 1,000 | credit_good | sex | logistic_regression | 0.712 |
| compas_two_years | 7,214 | two_year_recid | race | logistic_regression | 0.682 |
| communities | 1,994 | high_violent_crime | majority_white | logistic/random_forest | 0.850 |
| australian | 690 | credit_approved | gender_like | logistic_regression | 0.844 |

New: Communities and Crime (UCI) and Statlog Australian Credit (UCI) added via
old direct download URLs. `data.py` `add_age_group()` fixed to return df
unchanged when no `age` column exists.

### Accuracy estimation (new)

`scripts/estimate_accuracy.py` programmatically estimates hold-out accuracy for every
dataset via stratified 5-fold cross-validation + a tuned hold-out measurement
(`--tune` runs a small random search over LR / RF / gradient boosting grids).
No manual estimates anywhere. Results on the eight real datasets (logistic regression,
random forest, gradient boosting):

| Dataset | Best model | CV accuracy | Hold-out accuracy |
|---|---|---|---|
| demo_loans (synthetic, signal strengthened) | gradient_boosting | 0.985 ± 0.003 | **0.992** |
| student_performance | random_forest | 0.925 ± 0.009 | 0.900 |
| bank_marketing | gradient_boosting | 0.887 ± 0.002 | 0.888 |
| adult | gradient_boosting | 0.845 ± 0.003 | 0.846 |
| taiwan_credit | gradient_boosting | 0.823 ± 0.004 | 0.820 |
| heart | random_forest | 0.806 ± 0.032 | 0.855 |
| german | logistic_regression | 0.708 ± 0.005 | 0.712 |
| compas_two_years | logistic_regression | 0.677 ± 0.009 | 0.682 |

Only the synthetic demo data reaches the 0.99 target — on real tabular credit/banking
data 0.99 hold-out accuracy is not achievable without data leakage (the program
reports this honestly; best real result is student_performance at 0.900).

### Note on SHAP speed

`communities` has 99 encoded features; SHAP's PermutationExplainer takes ~3.5 min
on that dataset. All other datasets assess in seconds.

Pull requests are the next phase (as requested): each step becomes one branch
(`step-XX-name`) replayed from its local commit, with a detailed PR description,
opened against `Meet6338-X/FairGuard`.
