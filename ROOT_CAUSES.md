# Root-cause analysis — FairGuard AI

Environment: Python 3.12.6, sklearn 1.9.0, shap 0.54.1, fairlearn 0.15.0,
pandas 2.3.3, numpy 2.2.4. Windows. Repo: C:\Users\shahm\Projects\FairGuard

**Status: all 24 root causes fixed and covered by tests.**
`python -m pytest tests -q` → **125 passed**.
Baseline before the fixes was 59 tests; the audit added 66 more.

| # | Severity | Area | One-line summary |
| --- | --- | --- | --- |
| [RC-01](#rc-01--shap-always-used-the-slow-permutation-explainer--critical-fixed) | Critical | Explainability | SHAP always fell back to the 67× slower permutation explainer |
| [RC-02](#rc-02--assessments-were-scored-on-data-the-model-was-trained-on-traintest-contamination) | Critical | ML / evaluation | Assessments scored the model on its own training rows |
| [RC-03](#rc-03--pick_features-silently-dropped-every-categorical-column) | High | Features | Every categorical column was dropped from the model |
| [RC-04](#rc-04--fairness-risk-ignored-equalized-odds) | High | Risk | Equalized-odds violations were reported as "no blocking issues" |
| [RC-05](#rc-05--risk-score-was-not-comparable-between-assessments) | High | Risk | The same finding scored 0.18 in one run and 0.2769 in another |
| [RC-06](#rc-06--drift_risk-averaged-over-every-feature-and-diluted-critical-drift) | High | Drift | One critical feature in major drift averaged away to a risk of 0.05 |
| [RC-07](#rc-07--drift-was-always-measured-against-itself-false-assurance) | High | Drift | The dataset was compared with itself, so every PSI was 0 |
| [RC-08](#rc-08--privacy-scanner-produced-false-positives-on-numeric-identifier-columns) | Medium | Privacy | A 16-digit `account_id` was reported as a credit card (severity 1.0) |
| [RC-09](#rc-09--datavalidate_dataset-crashed-on-mixed-type-labels) | Medium | Data | Validation crashed with `TypeError` on mixed-type labels |
| [RC-10](#rc-10--train_test_split-aborted-when-a-class-had-fewer-than-2-members) | Medium | ML | Training aborted when a class had a single member |
| [RC-11](#rc-11--fairness-crashed-when-the-sensitive-column-had-nan) | Medium | Fairness | A NaN in the sensitive column crashed the run |
| [RC-12](#rc-12--the-optional-transformer-judge-backend-was-broken-by-design) | Medium | LLM safety | Every judge was called as a text-classification model |
| [RC-13](#rc-13--the-pii-text-judge-was-never-used) | Medium | LLM safety | The PII judge was dead code |
| [RC-14](#rc-14--only-the-first-judges-backend-was-reported) | Medium | LLM safety | A silent fallback to heuristics was invisible |
| [RC-15](#rc-15--heuristic-toxicity-risk-depended-on-text-length) | Medium | LLM safety | The toxicity score depended on the length of the text |
| [RC-16](#rc-16--apireview-ignored-run_id) | Medium | API | The review endpoint overwrote the newest report regardless of `run_id` |
| [RC-17](#rc-17--no-probability-calibration-check) | Medium | ML | No check that the predicted probabilities mean anything |
| [RC-18](#rc-18--no-statistical-uncertainty-on-fairness-numbers) | Medium | Fairness | Point estimates only — a 0.30 DPD on 303 rows looked like a fact |
| [RC-19](#rc-19--no-class-imbalance-handling) | Low | ML | No class weighting for imbalanced targets |
| [RC-20](#rc-20--apppy-showed-the-wrong-dataset-profile-after-auto-switch) | Low | UI | The Overview tab profiled the wrong dataframe |
| [RC-21](#rc-21--the-calibration-error-used-accuracy-instead-of-the-empirical-positive-rate-found-by-the-new-tests) | Medium | ML | A perfectly calibrated model scored an ECE of 0.5 |
| [RC-22](#rc-22--the-drift-self-reference-check-compared-the-wrong-frame-found-by-the-new-tests) | Low | Drift | The self-reference warning could never fire |
| [RC-23](#rc-23--a-label-duplicate-in-adultcsv-made-the-model-read-the-answer) | Critical | ML / leakage | `adult.csv` ships the raw label next to the derived one — after RC-03 the model trained on the answer and scored a bogus 1.000 |
| [RC-24](#rc-24--the-fairness-bootstrap-iterated-over-rows-not-groups) | High | Performance | The bootstrap was O(rows²) per draw — a 12 000-row hold-out took 10 minutes for the fairness step alone |
| [RC-23](#rc-23--a-label-duplicate-in-adultcsv-made-the-model-read-the-answer) | Critical | ML / leakage | `adult.csv` ships `income` next to `income_gt_50k`; after RC-03 the model trained on the answer (hold-out accuracy 1.000) |

## Verified root causes (each confirmed by an executed probe, not by reading)

### RC-01 — SHAP always used the slow permutation explainer  (CRITICAL, fixed)
`fairguard/explainability.py::_explainer` chose the explainer with
`isinstance(model, LogisticRegression)` / `isinstance(model, RandomForestClassifier)`.
Every FairGuard model is an sklearn `Pipeline` (imputer -> scaler -> estimator),
so **neither check ever matched** and every model fell through to
`shap.Explainer(model.predict_proba, ...)` = PermutationExplainer.
Measured on `communities.csv` (99 encoded features, 1994 rows):

    PermutationExplainer (old) : 234.5 s
    TreeExplainer      (new)   :   3.5 s     (~67x faster, and exact)

Also `linear` explainer was never reachable, so the demo logistic regression
took 2.49 s where 0.01 s suffices.
FIX: unwrap the Pipeline (`split_pipeline`), use `shap.LinearExplainer` for
linear models, `shap.TreeExplainer` for tree ensembles, permutation only as
fallback. Report the explainer kind in the assessment record.
Verified after fix: lr 0.01 s / rf 0.06 s / gb 0.07 s on the demo data.

### RC-02 — Assessments were scored on data the model was trained on (train/test contamination)
`train.train_model` fits on 75 % of the data; `scripts/run_assessment.py`,
`api.py` and `app.py` then assess the model on the **full** dataset.
Metrics are therefore in-sample. Effect is small for low-capacity models
(german: -0.001) but it is a methodology error that inflates any high-capacity
model, and it makes "accuracy" non-comparable with a true hold-out number.
FIX: hold out an evaluation split at training time, persist it in the model
artifact, and let the assessment report its `evaluation_scope`.

### RC-03 — `pick_features` silently dropped every categorical column
`train.pick_features` = `df.select_dtypes(include="number")` only.
On `adult.csv` that drops 9 categorical columns (workclass, education,
marital_status, occupation, relationship, race, native_country, ...) — a large
share of the predictive signal, and it also hides proxy variables for the
sensitive attribute (a fairness-monitoring blind spot).
FIX: keep low-cardinality categorical columns and one-hot encode them.

### RC-04 — Fairness risk ignored equalized odds
`risk.build_recommendations` only checked `demographic_parity_difference`.
A model with DPD 0.01 and EOD 0.30 got the recommendation
"No blocking issues found" (verified).
FIX: check both, and drive the text from `config` instead of the literal "0.70".

### RC-05 — Risk score was not comparable between assessments
`risk.compute_risk` normalises over the dimensions that were assessed.
Same fairness risk 0.6 → score 0.18 "Low" with 5 dimensions, 0.2769 "Medium"
with 3 dimensions (verified). A reviewer comparing two runs sees different
levels for the same underlying finding.
FIX: report `coverage` + `dimensions_assessed` next to the score.

### RC-06 — `drift_risk` averaged over every feature and diluted critical drift
Mean per-feature severity: 1 feature in MAJOR drift out of 20 → risk 0.05
(verified) — i.e. a critical feature drift reads as "no risk".
FIX: importance-weighted aggregation with a max-aware term.

### RC-07 — Drift was always measured against itself (false assurance)
`api.py` (`use_reference: bool = True`) and `app.py` pass `reference=df`, so
every PSI is exactly 0.0 and the Drift tab shows "stable" for everything
(verified: reference==current -> psi 0.0; reference==drifted -> psi 0.3037,
level "major").
FIX: default to a real reference, and label a self-reference baseline
explicitly so it cannot be read as "no drift".

### RC-08 — Privacy scanner produced false positives on numeric identifier columns
`privacy.PATTERNS["credit_card"] = r"\b(?:\d[ -]?){13,16}\b"` and
`["phone"] = r"(?:\+91[\s-]?)?[6-9]\d{9}\b"` matched a 16-digit `account_id`
as *both* "phone" and "credit_card" (verified: severity 1.0 on a numeric ID).
FIX: type-aware matching — numeric columns need a matching column-name trigger
before a PII pattern counts.

### RC-09 — `data.validate_dataset` crashed on mixed-type labels
`values <= {0, 1}` then `sorted(values)[:5]` → `TypeError: '<' not supported
between instances of 'str' and 'int'` (verified). String labels ("yes"/"no")
are reported as "not binary" even though they are a valid binary target.
FIX: robust binary detection that also accepts string/bool labels.

### RC-10 — `train_test_split` aborted when a class had fewer than 2 members
`ValueError: The least populated classes in y have only 1 member` (verified).
FIX: fall back to an unstratified split with a clear warning.

### RC-11 — Fairness crashed when the sensitive column had NaN
`MetricFrame` raised `ValueError: Feature 'gender' contains missing values`
(verified).
FIX: drop the missing rows, and report how many were dropped.

### RC-12 — The optional transformer judge backend was broken by design
`judges.TransformerJudge` always used `pipeline("text-classification", ...)`:
  * `iiiorg/piiranha-v1-detect-personal-information` is a **token-classification**
    model — the wrong task entirely;
  * `vectara/hallucination_evaluation_model` is a **cross-encoder** that needs the
    reference as a second input.
FIX: per-kind HF task in `config.JUDGE_TASKS`.

### RC-13 — The PII text judge was never used
`llm_eval.DEFAULT_KINDS = ("toxicity", "bias", "hate_speech")` — the `pii`
judge existed but was dead code (verified: kinds evaluated are only the three).
FIX: include `pii` by default.

### RC-14 — Only the first judge's backend was reported
`llm_eval.evaluate_texts` returned `judges[kinds[0]].name`, so a silent
per-kind fallback to the heuristic backend was invisible in the report.
FIX: report per-kind backends.

### RC-15 — Heuristic toxicity risk depended on text length
Same content, different score: `"stupid"` → 1.0, `"stupid ok ok ... (41 words)"`
→ 0.122 (verified). `_word_risk` divides by word count.
FIX: length-bucketed normalisation.

### RC-16 — `api.review` ignored `run_id`
`api.py::review` picks the **newest** report file by mtime and ignores
`req.run_id` entirely (verified: 0 occurrences of `req.run_id` in the function
body) — a reviewer decision can be attached to the wrong assessment.
FIX: look the run up by id.

### RC-17 — No probability calibration check
Only accuracy/precision/recall/F1/ROC-AUC. An over-confident model can look
excellent while its probabilities are meaningless for a threshold-based
decision. FIX: add Brier score + expected calibration error.

### RC-18 — No statistical uncertainty on fairness numbers
Point estimates only. With small groups (heart: 303 rows) a DPD of 0.30 may be
noise. `fairlearn.metrics.MetricFrame` supports bootstrap CIs
(`n_boot`, `ci_quantiles`, `random_state` — verified available in 0.15.0).
FIX: bootstrap CIs + flag groups below `config.MIN_GROUP_SIZE`.

### RC-19 — No class-imbalance handling
`make_model` used default weights; on imbalanced targets (taiwan_credit
default ~22 %) recall for the minority class suffers, which shows up directly
in equalized odds. FIX: `class_weight="balanced"` option.

### RC-20 — `app.py` showed the wrong dataset profile after auto-switch
`run_pipeline` may switch `df` to the model's dataset, but the Overview tab
renders the outer `df` (the pre-switch frame). FIX: render the assessed frame.

### RC-21 — The calibration error used accuracy instead of the empirical positive rate (found by the new tests)
`performance.expected_calibration_error` compared, per bin, the fraction of
*predictions* that were correct with the mean *confidence*. That is not the
definition of calibration error. With `y_true=[1,1,0,0]` and
`y_prob=[1,1,0,0]` — a perfectly calibrated model — it returned **0.5**
instead of **0.0** (verified by `tests/test_calibration.py`).
FIX: per bin, compare `mean(y_true[mask])` (the empirical positive rate) with
`mean(y_prob[mask])` (the mean confidence), then weight by the bin's share of
the sample. A perfectly calibrated model now scores 0.

### RC-22 — The drift self-reference check compared the wrong frame (found by the new tests)
`pipeline.run_assessment` compared `reference is df` to decide whether the
reference was the assessed dataset — but `df` had already been re-assigned by
`add_age_group` / `encode_label`, so the identity check never fired and the
"this is a baseline, not evidence of stability" note was never shown.
FIX: capture the original frame as `assessed` before re-encoding and compare
`reference is assessed`.

### RC-23 — A label duplicate in adult.csv made the model read the answer
(CRITICAL, fixed) Found while re-measuring the accuracy table for the README:
`data/adult.csv` hold-out accuracy was **1.000** — a model that never misses
on census data is a red flag, not a triumph.

`adult.csv` carries the raw label column `income` (values `">50K"` / `"<=50K"`)
next to its derived label `income_gt_50k`. RC-03's fix deliberately **keeps**
low-cardinality categorical columns, so `income` became a feature and the
gradient boosting model read the answer straight off it.

The check is mechanical: a column whose distinct values map **bijectively**
onto the label's values (every value of the column determines the label, and
vice versa) is a renamed target. Scanned all eight real datasets with a probe:

| Dataset | Label | Leak found |
| --- | --- | --- |
| adult | `income_gt_50k` | **`income`** |
| heart, compas_two_years, taiwan_credit, australian, german, communities, bank_marketing, student_performance | — | none |

FIX in `fairguard/train.py`:

1. `label_duplicate_columns(df, label, max_categories=50)` returns every
   bijective (1:1) duplicate of the label — columns with more than 50 distinct
   values are ignored, because a high-cardinality identifier is noise rather
   than an inversion, and dropping every unique-id column would gut the model.
2. `pick_features` adds that set to the skip list.
3. `train_model` / `make_holdout` also filter an **explicitly passed** feature
   list through the same guard — a caller cannot quietly leak through it.

Measured effect: adult hold-out accuracy **0.829** (was 1.000), risk 0.4615
Medium / "Approve with monitoring". `tests/test_leakage_guard.py` pins all of
this down, including the explicit-feature-list path.

**Defence in depth.** A guard only at training time leaves every artifact
saved *before* the fix silently dangerous, so RC-23 is also enforced where
the score is produced:

1. `pipeline.run_assessment` re-runs the same leak test at assessment time.
   If the artifact's `raw_features` still contain the label or a 1:1 copy of
   it, the record gets an `assessment["leakage"]` block
   (`{"columns": [...], "note": "…"}`) and `evaluation.in_sample` is forced
   to True.
2. `risk.build_recommendations` puts "Leakage: the model uses <cols> which
   are 1:1 copies of the label…" **first** in the recommendation list — it
   invalidates everything else on the page.
3. `report.build_markdown` / `build_model_card` render a "⚠️ Label leakage
   detected" section, so nobody can read the accuracy number without walking
   past the warning. `tests/test_leak_assessment.py` covers all four cases
   (leak via features, leak via the label itself, the perfect-score
   stale-artifact scenario, and the clean case).

**Artifact refresh.** `scripts/retrain_all.py` retrains every
dataset × model type with the current train.py (hold-out + balanced weights +
one-hot categoricals + leak guard). One run refreshed 30 artifacts with 0
errors; the audit then confirmed **34/34 artifacts** in models/ now carry a
stored hold-out and no leak features.

### RC-24 — The fairness bootstrap iterated over rows, not groups (found by profiling the slow Adult run)
**Severity: High, fixed.** After the earlier fixes the Adult reveal was still
painfully slow: the CLI assessment sat for >10 minutes. A per-step profile
(`imp 2.09 s · load 2.19 s · performance 0.02 s · **fairness 639.24 s** ·
privacy 1.40 s · explainability 0.30 s`) put the entire cost in one place:
`fairness.bootstrap_intervals` on a 12 211-row hold-out.

`fairness._per_group_rates` wrote:

```python
for group in groups:                      # groups is an array of row labels
```

Iterating a numpy array walks **every row**, so for each row the code
re-scanned the whole vector with `groups == group` — O(rows²) per bootstrap
draw, × 200 draws ≈ 29.8 billion comparisons per assessment. The demo's
500-row hold-out finished in a couple of seconds, which is why the test suite
never noticed.

FIX (one line): walk the distinct groups, not the rows —

```python
unique = np.unique(groups)
for group in unique:
```

Measured after the fix: bootstrap on 12 000 rows **0.86 s** (previously >900 s,
timeout), identical output; verified for numeric, categorical, bool and string
group columns. `python scripts/run_assessment.py --data data/adult.csv ...`
fell from 644 s to **9.4 s** end to end. Regression test in
`tests/test_bootstrap_scale.py` (includes a 12 000-row guard that must finish
in under 5 s).

## Files changed
| File | Root causes addressed |
| --- | --- |
| `fairguard/config.py` | new thresholds, `EVAL_SPLIT`, `PII_NAME_TRIGGERS`, `JUDGE_TASKS`, `MIN_F1`, `CALIBRATION_BINS` |
| `fairguard/data.py` | RC-09 |
| `fairguard/train.py` | RC-02, RC-03, RC-10, RC-19, RC-23 |
| `fairguard/performance.py` | RC-17, RC-21 |
| `fairguard/fairness.py` | RC-11, RC-18, RC-24 |
| `fairguard/explainability.py` | RC-01 |
| `fairguard/privacy.py` | RC-08 |
| `fairguard/drift.py` | RC-06 |
| `fairguard/risk.py` | RC-04, RC-05 |
| `fairguard/judges.py` | RC-12, RC-15 |
| `fairguard/llm_eval.py` | RC-13, RC-14 |
| `fairguard/pipeline.py` | RC-02, RC-05, RC-07, RC-22 |
| `fairguard/report.py` | evaluation scope / coverage / CIs / calibration sections, report-name collisions |
| `fairguard/tracking.py` | `scope` + `explainer` tags, CI + coverage metrics |
| `scripts/train_demo_model.py` | RC-02 (stores the hold-out), `--test-size`, `--no-balance` |
| `scripts/run_assessment.py` | RC-02 (scores the hold-out), RC-07 (reference defaults to none) |
| `api.py` | RC-07, RC-16 (`run_id` honoured), RC-02 |
| `app.py` | RC-02, RC-07, RC-20 |

## Tests added
| File | Covers |
| --- | --- |
| `tests/test_fairness_ci.py` | RC-18, RC-11, RC-04 |
| `tests/test_explainability_kinds.py` | RC-01 |
| `tests/test_calibration.py` | RC-17, RC-21 |
| `tests/test_holdout_scope.py` | RC-02 |
| `tests/test_privacy_types.py` | RC-08 |
| `tests/test_api_review.py` | RC-16, RC-07 |
| `tests/test_drift.py` (extended) | RC-06 |
| `tests/test_llm_eval.py` (extended) | RC-13, RC-14 |
| `tests/test_leakage_guard.py` | RC-23 |
| `tests/test_bootstrap_scale.py` | RC-24 |
| `tests/test_leak_assessment.py` | RC-23 (assessment-time guard) |

## Verification

```
python -m pytest tests -q                      -> 125 passed in 15s
python scripts/make_demo_data.py               -> demo_loans.csv (2000) + demo_loans_new.csv (2000, drifted)
python scripts/train_demo_model.py             -> lr 0.948 / rf 0.990 / gb 0.992 hold-out accuracy
python scripts/run_assessment.py --sensitive gender
                                              -> scope holdout (500 rows), risk 0.5298 High, coverage 65%
python scripts/run_assessment.py --reference data/demo_loans_new.csv --model models/gradient_boosting.pkl
                                              -> scope holdout, risk 0.7335 High, coverage 85%, EOD 0.38 + drift flagged
API smoke test                                -> /api/assess returns scope=holdout + CIs + explainer=linear,
                                                 /api/review writes to the requested run_id only
```

