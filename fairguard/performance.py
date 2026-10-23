"""Conventional performance evaluation — the baseline every assessment starts from.

Root cause fixed here
----------------------
* **RC-17** Only discrimination metrics were reported.  A model can be
  accurate while its *probabilities* are badly miscalibrated, which matters
  because the workbench makes threshold-based decisions.  Brier score and the
  expected calibration error are now reported as well.
* The confusion matrix is always built with ``labels=[0, 1]`` so its shape is
  stable even when a split happens to contain a single class (this used to
  emit a sklearn warning).
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config


def expected_calibration_error(y_true, y_prob, n_bins=None) -> float:
    """Expected Calibration Error — mean |empirical rate - confidence| per bin.

    For every probability bin the *empirical positive rate* (the share of
    rows whose label is 1) is compared with the *average predicted
    probability* in that bin.  A calibrated model has the two equal in every
    bin, so the error is 0.

    0.0 means every probability bin behaves exactly as its confidence
    promises; 1.0 means the probabilities are worthless.
    """
    n_bins = n_bins or config.CALIBRATION_BINS
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize with right=True puts a value of exactly 1.0 in the last bin.
    bins = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)
    error = 0.0
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        # The empirical rate of positives in the bin, not the classification
        # accuracy: calibration asks "do 70 %-confidence rows turn out to be
        # positives 70 % of the time?", not "how often was the class right?".
        positive_rate = float(np.mean(y_true[mask]))
        confidence = float(np.mean(y_prob[mask]))
        error += (mask.sum() / y_true.size) * abs(positive_rate - confidence)
    return float(error)


def evaluate_performance(y_true, y_pred, y_prob=None) -> dict:
    """Accuracy / precision / recall / F1 (+ ROC-AUC and calibration when probabilities are given)."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None and len(set(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    elif y_prob is not None:
        metrics["roc_auc"] = None
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    if y_prob is not None:
        metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
        metrics["calibration_error"] = round(expected_calibration_error(y_true, y_prob), 4)
    return metrics


def performance_risk(metrics: dict, min_accuracy=None, min_f1=None) -> float:
    """0..1 risk from the accuracy/F1 shortfall (0 = at/above the floor, 1 = far below).

    Accuracy and F1 are considered together: a model that is accurate only by
    predicting the majority class scores badly on F1.
    """
    floor = min_accuracy if min_accuracy is not None else config.MIN_ACCURACY
    f1_floor = min_f1 if min_f1 is not None else config.MIN_F1
    accuracy = metrics.get("accuracy", 1.0)
    f1 = metrics.get("f1", 1.0)
    accuracy_gap = max(0.0, floor - accuracy) / floor
    f1_gap = max(0.0, f1_floor - f1) / f1_floor
    return float(min(max(accuracy_gap, f1_gap), 1.0))
