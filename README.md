# FairGuard AI — Responsible AI Assessment Workbench

**Team Gear5thCoders · Group 119 · Vishwakarma Institute of Technology, Pune**

FairGuard evaluates machine-learning models *beyond accuracy*. It measures fairness,
explainability, privacy risk, data/model drift, performance and text/LLM safety,
combines the results into one risk score with actionable recommendations, records
every assessment in MLflow, and presents everything in an interactive Streamlit
dashboard with an audit-ready report and an automated model card.

| Dimension | What it checks | Method |
|---|---|---|
| Fairness | Group-wise metrics and disparity across sensitive attributes | Fairlearn |
| Explainability | Global feature importance + per-prediction explanation | SHAP |
| Privacy | PII / sensitive columns and privacy-risk summary | Detectors (+ optional model) |
| Drift | Distribution shift between reference and current data | PSI + KS test |
| Performance | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix | scikit-learn |
| Text / LLM safety | Toxicity, bias, hate speech, hallucination, PII in text | HF classifiers or built-in heuristics |
| Risk | Weighted risk score Low → Critical, recommendations, deployment decision | Threshold engine |
| Tracking | Every run logged with params, metrics and artifacts | MLflow |

## How an assessment flows

```
dataset + trained model + sensitive attributes
        │
        ▼
  data.py  (load, validate, profile)
        │
        ▼
  assessment pipeline (pipeline.py)
   ├── performance.py     accuracy / F1 / ROC-AUC / confusion matrix
   ├── fairness.py        group metrics, demographic parity, equalized odds
   ├── explainability.py  SHAP global + local
   ├── privacy.py         PII / sensitive columns
   ├── drift.py           PSI + KS vs reference data
   └── judges.py / llm_eval.py   text safety (optional)
        │
        ▼
  risk.py (score + recommendations) → report.py (JSON + Markdown + model card)
        │
        ▼
  tracking.py (MLflow) → app.py (Streamlit dashboard)
```

## Quickstart

```bash
python -m pip install -r requirements.txt

# 1. synthetic demo dataset (biased, with PII columns and a drifted version)
python scripts/make_demo_data.py

# 2. train two baseline models into models/
python scripts/train_demo_model.py

# 3. run a full assessment from the command line
python scripts/run_assessment.py --data data/demo_loans.csv --model models/logistic_regression.pkl --sensitive gender

# 4. or open the interactive dashboard
streamlit run app.py
```

## Datasets

The demo runs on a synthetic loan CSV the project generates itself. For real
data, four open datasets from the UCI ML Repository and ProPublica download
with one command (no account or API key needed):

```
python scripts/download_datasets.py
```

| File | Label column | Sensitive column | Source |
|---|---|---|---|
| `data/adult.csv` | `income_gt_50k` | `sex` (also `race`, `age`) | UCI — Adult / census income |
| `data/german.csv` | `credit_good` | `sex` | UCI — Statlog German credit |
| `data/taiwan_credit.csv` | `default_next_month` | `gender` | UCI — credit card default |
| `data/compas_two_years.csv` | `two_year_recid` | `race` (also `sex`) | ProPublica — COMPAS recidivism |
| `data/bank_marketing.csv` | `subscribed` | `age_group` (also `age`) | UCI — bank marketing |
| `data/student_performance.csv` | `G3_pass` | `sex` | UCI — student performance |
| `data/heart.csv` | `disease` | `sex` | UCI — Cleveland heart |
| `data/communities.csv` | `high_violent_crime` | `majority_white` | UCI — Communities and Crime |
| `data/australian.csv` | `credit_approved` | `gender_like` | UCI — Statlog Australian Credit |

Example on real data (files are git-ignored; regenerate any time, personal
name columns are dropped by the converter):

```
python scripts/train_demo_model.py --data data/adult.csv --label income_gt_50k --sensitive sex
python scripts/run_assessment.py --data data/adult.csv --label income_gt_50k --sensitive sex --model models/adult_logistic_regression.pkl
```

## Models

Every dataset gets three model types — `logistic_regression` (median-imputed
and standardized), `random_forest`, and `gradient_boosting` (sklearn
histogram trees), all wrapped in sklearn pipelines so real CSVs with stray
`?`/NaN values train and predict without extra preparation:

```
python scripts/train_demo_model.py --data data/bank_marketing.csv --label subscribed --sensitive sex --model-type gradient_boosting
```

Best test-split accuracy per dataset (24 models fully trained):

| Dataset | Best model | Accuracy |
|---|---|---|
| student_performance | logistic_regression | 0.927 |
| heart | gradient_boosting | 0.868 |
| bank_marketing | gradient_boosting | 0.888 |
| adult | gradient_boosting | 0.846 |
| taiwan_credit | gradient_boosting | 0.820 |
| german | logistic/gradient_boosting | 0.712 |
| communities | logistic/random_forest | 0.850 |
| australian | logistic_regression | 0.844 |
| demo loans | gradient_boosting | 0.992 |
| compas_two_years | logistic_regression | 0.682 |

The bank marketing converter drops the `duration` column on purpose — it
leaks the target (call length is known only after the call), a documented
trap of that dataset.

## Results on real data

Assessing the best model per dataset (full pipeline: fairness + SHAP +
privacy + drift + report):

- Adult risk Medium / `Approve with monitoring` · Taiwan Low / `Approve` ·
  German High / `Needs review` · COMPAS High / `Needs review` with a large
  racial recidivism gap — the classic ProPublica finding, reproduced by the
  tool. Heart disease: High / `Needs review` with a 0.30 sex-based
  diagnosis-rate gap that fairness scores catch immediately.
- Communities and Crime: High / `Needs review` — violent-crime rate differs
  sharply by racial-majority group (dpd ≈ 0.65), acc ≈ 0.86.
- Australian credit: Medium / `Approve with monitoring` — credit approval
  differs strongly by gender-like attribute (dpd ≈ 0.49), acc ≈ 0.87.

## Project layout

```
fairguard/        assessment modules (one responsibility per file)
scripts/          demo data, demo training, CLI assessment
tests/            unit + integration tests
app.py            Streamlit dashboard
data/, models/    generated at runtime (not committed)
mlruns/           MLflow tracking store (local sqlite db, created on first run)
reports/          exported assessment reports (created on first run)
```

## Notes

- No paid APIs or API keys are required. Text judges run locally through Hugging
  Face `transformers` when it is installed; otherwise lightweight heuristic
  judges are used, so the workbench always runs offline.
- Fairness results are statistical measurements of *this model on this dataset*.
  They are not conclusions about protected groups of people.

## Project info

- Course: Project 23 — Responsible AI Bias, Explainability & Privacy Assessment Workbench
- Guide: Mr. Kishor Pathak
- Members: Meet Shah (2), Vaishnavi Pawar (18), Purva Waykole (20), Dipak Pote (32)
