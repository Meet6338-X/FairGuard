"""Conventional performance evaluation — the baseline every assessment starts from."""
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config


def evaluate_performance(y_true, y_pred, y_prob=None) -> dict:
    """Accuracy / precision / recall / F1 (+ ROC-AUC when probabilities are given)."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None and len(set(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return metrics


def performance_risk(metrics: dict, min_accuracy=None) -> float:
    """0..1 risk from the accuracy shortfall (0 = at/above the floor, 1 = far below)."""
    floor = min_accuracy if min_accuracy is not None else config.MIN_ACCURACY
    gap = floor - metrics["accuracy"]
    if gap <= 0:
        return 0.0
    return float(min(gap / floor, 1.0))
