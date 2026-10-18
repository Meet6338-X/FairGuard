"""Shared fixtures. Everything here runs offline and fast."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fairguard import train  # noqa: E402
from scripts.make_demo_data import make  # noqa: E402


@pytest.fixture(scope="session")
def demo_df():
    """Small biased lending dataset (400 rows)."""
    return make(n=400, seed=1)


@pytest.fixture(scope="session")
def drifted_df():
    """Same dataset, but shifted as if the input population changed."""
    return make(n=400, seed=1, drift=True)


@pytest.fixture(scope="session")
def bundle(demo_df):
    """A trained logistic-regression model plus its train/test split."""
    model, X_train, X_test, y_train, y_test = train.train_model(
        demo_df, "logistic_regression"
    )
    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


@pytest.fixture(scope="session")
def sample_assessment():
    """A hand-built assessment record for report/tracking tests."""
    return {
        "model_id": "demo_model",
        "model_version": "1.0",
        "dataset_id": "demo_data",
        "sensitive_attribute": "gender",
        "threshold": 0.5,
        "assessed_at": "2026-10-17T10:30:00+00:00",
        "performance": {
            "accuracy": 0.81,
            "precision": 0.74,
            "recall": 0.66,
            "f1": 0.70,
            "roc_auc": 0.86,
            "confusion_matrix": [[60, 15], [22, 43]],
        },
        "fairness": {
            "sensitive_attribute": "gender",
            "by_group": {
                "selection_rate": {"female": 0.42, "male": 0.63},
                "accuracy": {"female": 0.78, "male": 0.83},
                "precision": {"female": 0.70, "male": 0.76},
                "recall": {"female": 0.60, "male": 0.70},
            },
            "demographic_parity_difference": 0.21,
            "equalized_odds_difference": 0.10,
            "group_sizes": {"female": 140, "male": 260},
        },
        "privacy": {
            "findings": [
                {"column": "email", "type": "email", "matches": 120, "severity": 0.9,
                 "samples": ["me****"]},
            ],
            "risk": 0.27,
        },
        "explainability": {
            "global_importance": {"credit_score": 0.42, "age": 0.18, "income_lakhs": 0.11}
        },
        "drift": {
            "income_lakhs": {"psi": 0.31, "psi_level": "major", "ks_stat": 0.35,
                             "ks_pvalue": 0.0, "missing_pct_shift": 0.0},
            "gender": {"psi": 0.01, "psi_level": "stable", "ks_stat": None,
                       "ks_pvalue": None, "missing_pct_shift": 0.0},
        },
        "text": {
            "backend": "heuristic", "n_texts": 2, "risk": 0.1, "flagged": 0,
            "kind_summary": {"toxicity": 0.1, "bias": 0.05, "hate_speech": 0.0},
        },
        "risk": {
            "score": 0.42,
            "level": "Medium",
            "contributions": {"fairness": 0.15, "privacy": 0.08, "drift": 0.10,
                              "performance": 0.02, "text": 0.07},
        },
        "recommendations": ["Do a thing.", "Keep monitoring."],
        "decision": "Approve with monitoring",
        "reviewer_decision": None,
    }
