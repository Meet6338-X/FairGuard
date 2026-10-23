"""End-to-end assessment — one call runs every dimension and returns one record.

Root causes fixed here
-----------------------
* **RC-02** the assessment used to score the model on *the whole dataset*,
  including the rows it was trained on.  A trained model now carries its
  hold-out split (``train.make_holdout``) and the assessment can be run on
  that hold-out, so "accuracy" means out-of-sample accuracy.  The record
  states which scope was used.
* **RC-07** when the reference dataset *is* the assessed dataset every PSI
  is exactly 0.0 — that is a baseline, not evidence that nothing drifted.
  The record now says so explicitly instead of showing an all-green tab.
"""
from datetime import datetime, timezone

from . import (
    data,
    explainability,
    llm_eval,
    performance,
    privacy,
    risk,
    tracking,
    train,
)
from . import drift as drift_mod
from . import fairness as fairness_mod


def run_assessment(
    df,
    model,
    features,
    label="approved",
    sensitive="gender",
    reference=None,
    model_id="model",
    model_version="1.0",
    dataset_id="dataset",
    threshold=0.5,
    include_explainability=True,
    run_mlflow=True,
    extra_artifacts=None,
    text=None,
    raw_features=None,
    eval_df=None,
    n_boot=None,
) -> dict:
    """Assess one trained model on one dataset and return the full record.

    df         dataset to assess (model inputs + label + sensitive column)
    model      fitted sklearn estimator with predict_proba
    features   encoded feature column list the model was trained on
    raw_features  optional pre-encoding columns (pass artifact['raw_features']
                  for real datasets; defaults to the demo FEATURES behavior)
    reference  reference dataset for drift (drift is skipped when None)
    text       optional pre-computed text assessment (llm_eval.evaluate_texts)
    eval_df    optional *hold-out* frame to score performance/fairness on
               (RC-02).  When omitted the whole dataset is used and the
               record says so via ``evaluation.scope``.
    """
    assessed = df                       # keep the original identity for the
    df = data.add_age_group(df)         # self-reference check in RC-07
    df, label_mapping = data.encode_label(df, label)
    problems = data.validate_dataset(df, [sensitive], label=label, model_features=features)
    if problems:
        raise ValueError("dataset validation failed: " + "; ".join(problems))

    # RC-02: metrics come from the hold-out when one is supplied.  The
    # hold-out is encoded independently so it may be a different frame.
    metrics_df = df
    if eval_df is not None:
        metrics_df = data.add_age_group(eval_df)
        metrics_df, _ = data.encode_label(metrics_df, label)
        if sensitive not in metrics_df.columns:
            raise ValueError(
                f"hold-out evaluation set is missing the sensitive column '{sensitive}'"
            )
        if label not in metrics_df.columns:
            raise ValueError(f"hold-out evaluation set is missing the label '{label}'")

    X = train.encode_features(df, features=raw_features, columns=features)
    X_eval = (X if eval_df is None
              else train.encode_features(metrics_df, features=raw_features, columns=features))
    y_true = metrics_df[label]
    y_prob = model.predict_proba(X_eval)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # RC-23 hardening: artifacts trained before the leak guard can still
    # carry a feature that maps 1:1 onto the label (adult.csv's `income`
    # copy of `income_gt_50k`).  Detect that *at assessment time* so a
    # stale artifact is flagged loudly instead of reporting ~1.0 accuracy.
    leak_columns = []
    if raw_features:
        # A feature equal to the label itself is a leak too.
        leak_candidates = set(train.label_duplicate_columns(assessed, label)) | {label}
        leak_columns = sorted(set(raw_features) & leak_candidates)

    assessment = {
        "model_id": model_id,
        "model_version": str(model_version),
        "dataset_id": dataset_id,
        "sensitive_attribute": sensitive,
        "threshold": threshold,
        "evaluation": {
            "scope": "holdout" if eval_df is not None else "full_dataset",
            "n_rows": int(len(metrics_df)),
            "n_dataset_rows": int(len(df)),
            "in_sample": eval_df is None or bool(leak_columns),
        },
        "performance": performance.evaluate_performance(y_true, y_pred, y_prob),
        "fairness": fairness_mod.assess_fairness(
            y_true, y_pred, metrics_df[sensitive], sensitive, n_boot=n_boot
        ),
        "privacy": privacy.scan_pii(df),
    }
    if label_mapping:
        assessment["label_mapping"] = {str(k): int(v) for k, v in label_mapping.items()}

    if leak_columns:
        assessment["leakage"] = {
            "columns": leak_columns,
            "note": ("these feature columns map 1:1 onto the label — every metric "
                     "below is in-sample knowledge, not generalisation"),
        }

    if include_explainability:
        importance = explainability.global_importance_with_meta(model, X_eval)
        assessment["explainability"] = importance

    if reference is not None:
        assessment["drift"] = drift_mod.assess_drift(reference, df)
        # RC-07: make a self-reference baseline impossible to mistake for
        # "the data did not drift".  The comparison uses the *original* frame
        # identity — `df` has been re-encoded by now.
        self_reference = reference is assessed
        assessment["drift_reference"] = {
            "same_as_assessed": bool(self_reference),
            "n_rows": int(len(reference)),
            "note": ("Reference is the assessed dataset itself - every PSI is 0 by "
                     "construction. This is a baseline, not evidence of stability."
                     if self_reference else
                     "Reference dataset differs from the assessed dataset."),
        }

    if text:
        assessment["text"] = text

    risks = {
        "performance": performance.performance_risk(assessment["performance"]),
        "fairness": fairness_mod.fairness_risk(assessment["fairness"]),
        "privacy": assessment["privacy"]["risk"],
    }
    if "drift" in assessment:
        importance = (assessment.get("explainability") or {}).get("global_importance")
        risks["drift"] = drift_mod.drift_risk(assessment["drift"], importance=importance,
                                              raw_features=raw_features)
    if "text" in assessment:
        risks["text"] = llm_eval.text_risk(assessment["text"])
    assessment["risk"] = risk.compute_risk(risks)
    assessment["recommendations"] = risk.build_recommendations(assessment)
    assessment["decision"] = risk.deployment_decision(assessment["risk"]["level"])
    assessment["reviewer_decision"] = None
    assessment["assessed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if run_mlflow:
        assessment["mlflow_run_id"] = tracking.log_assessment(assessment, extra_artifacts=extra_artifacts)
    return assessment


def compare_models(models: dict, df, **kwargs) -> dict:
    """Assess several models on the same dataset: {name: (model, features)} → {name: record}.

    Extra keyword arguments are forwarded to run_assessment.
    """
    results = {}
    for name, (model, features) in models.items():
        results[name] = run_assessment(df=df, model=model, features=features, model_id=name, **kwargs)
    return results
