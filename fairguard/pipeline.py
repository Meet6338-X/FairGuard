"""End-to-end assessment — one call runs every dimension and returns one record."""
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
) -> dict:
    """Assess one trained model on one dataset and return the full record.

    df         dataset to assess (model inputs + label + sensitive column)
    model      fitted sklearn estimator with predict_proba
    features   encoded feature column list the model was trained on
    raw_features  optional pre-encoding columns (pass artifact['raw_features']
                  for real datasets; defaults to the demo FEATURES behavior)
    reference  reference dataset for drift (drift is skipped when None)
    text       optional pre-computed text assessment (llm_eval.evaluate_texts)
    """
    df = data.add_age_group(df)
    problems = data.validate_dataset(df, [sensitive], label=label, model_features=features)
    if problems:
        raise ValueError("dataset validation failed: " + "; ".join(problems))

    X = train.encode_features(df, features=raw_features, columns=features)
    y_true = df[label]
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    assessment = {
        "model_id": model_id,
        "model_version": str(model_version),
        "dataset_id": dataset_id,
        "sensitive_attribute": sensitive,
        "threshold": threshold,
        "performance": performance.evaluate_performance(y_true, y_pred, y_prob),
        "fairness": fairness_mod.assess_fairness(y_true, y_pred, df[sensitive], sensitive),
        "privacy": privacy.scan_pii(df),
    }
    if include_explainability:
        assessment["explainability"] = {
            "global_importance": explainability.global_importance(model, X)
        }
    if reference is not None:
        assessment["drift"] = drift_mod.assess_drift(reference, df)
    if text:
        assessment["text"] = text

    risks = {
        "performance": performance.performance_risk(assessment["performance"]),
        "fairness": fairness_mod.fairness_risk(assessment["fairness"]),
        "privacy": assessment["privacy"]["risk"],
    }
    if "drift" in assessment:
        risks["drift"] = drift_mod.drift_risk(assessment["drift"])
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
