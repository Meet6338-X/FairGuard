"""MLflow tracking — every assessment becomes a documented experiment run."""
from pathlib import Path

import mlflow
import pandas as pd

from . import config

EXPERIMENT = "FairGuard Assessments"


def _point_mlflow_at(tracking_dir=None) -> Path:
    """Set the MLflow tracking URI to a local SQLite database (no server needed)."""
    location = Path(tracking_dir or config.MLRUNS_DIR)
    location.mkdir(parents=True, exist_ok=True)
    database = location / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{database.as_posix()}")
    return location


def log_assessment(assessment: dict, tracking_dir=None, extra_artifacts=None) -> str:
    """Log one assessment (params, metrics, artifacts) and return the run id."""
    _point_mlflow_at(tracking_dir)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"{assessment['model_id']} · {assessment['sensitive_attribute']}") as run:
        mlflow.log_params({
            "model_id": assessment["model_id"],
            "model_version": assessment["model_version"],
            "dataset_id": assessment["dataset_id"],
            "sensitive_attribute": assessment["sensitive_attribute"],
        })
        mlflow.set_tags({
            "risk_level": assessment["risk"]["level"],
            "decision": assessment["decision"],
            "reviewer_decision": assessment.get("reviewer_decision") or "pending",
        })

        metrics = {"risk_score": assessment["risk"]["score"]}
        for key, value in (assessment.get("performance") or {}).items():
            if isinstance(value, (int, float)):
                metrics[f"performance.{key}"] = float(value)
        fairness = assessment.get("fairness") or {}
        for key in ("demographic_parity_difference", "equalized_odds_difference"):
            if key in fairness:
                metrics[f"fairness.{key}"] = float(fairness[key])
        if (assessment.get("drift") or {}):
            metrics["drift.max_psi"] = max(entry["psi"] for entry in assessment["drift"].values())
        mlflow.log_metrics(metrics)

        mlflow.log_dict(assessment, "assessment.json")
        for path in extra_artifacts or []:
            if Path(path).exists():
                mlflow.log_artifact(str(path))

    return run.info.run_id


def history(tracking_dir=None) -> pd.DataFrame:
    """Past FairGuard runs as a tidy DataFrame (for the dashboard)."""
    _point_mlflow_at(tracking_dir)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        return pd.DataFrame()
    runs = mlflow.search_runs([experiment.experiment_id], order_by=["start_time DESC"])
    if runs.empty:
        return pd.DataFrame()
    columns = {
        "run_id": "run_id",
        "start_time": "started",
        "tags.mlflow.runName": "run",
        "params.model_id": "model",
        "params.model_version": "version",
        "params.sensitive_attribute": "sensitive",
        "metrics.risk_score": "risk",
        "tags.risk_level": "level",
        "tags.decision": "decision",
        "tags.reviewer_decision": "reviewer",
    }
    present = {key: label for key, label in columns.items() if key in runs.columns}
    history_frame = runs[list(present)].rename(columns=present)
    if "started" in history_frame.columns:
        history_frame["started"] = pd.to_datetime(history_frame["started"]).astype(str)
    return history_frame
