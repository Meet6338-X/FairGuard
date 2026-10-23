"""Tests for the calibration metrics — RC-17.

Only discrimination metrics (accuracy/precision/recall/F1/ROC-AUC) were
reported before.  A model can discriminate well while its *probabilities* are
badly miscalibrated, which matters because FairGuard makes threshold-based
decisions.
"""
import numpy as np

from fairguard import performance


def test_perfectly_calibrated_probabilities_have_zero_error():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([1.0, 1.0, 0.0, 0.0])
    assert performance.expected_calibration_error(y_true, y_prob) == 0.0


def test_confidently_wrong_probabilities_have_high_error():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.99, 0.99, 0.99, 0.99])
    assert performance.expected_calibration_error(y_true, y_prob) > 0.4


def test_evaluate_performance_reports_calibration_when_probabilities_given():
    metrics = performance.evaluate_performance(
        [1, 0, 1, 0], [1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2]
    )
    assert "brier_score" in metrics
    assert "calibration_error" in metrics
    assert 0 <= metrics["brier_score"] <= 1
    assert 0 <= metrics["calibration_error"] <= 1


def test_evaluate_performance_omits_calibration_without_probabilities():
    metrics = performance.evaluate_performance([1, 0, 1, 0], [1, 0, 1, 0])
    assert "brier_score" not in metrics
    assert "calibration_error" not in metrics
    assert "roc_auc" not in metrics


def test_confusion_matrix_shape_is_stable_for_single_class_splits():
    metrics = performance.evaluate_performance([1, 1, 1], [1, 1, 1])
    assert metrics["confusion_matrix"] == [[0, 0], [0, 3]]


def test_performance_risk_uses_the_f1_floor():
    """RC: a model that is accurate only by predicting the majority class."""
    metrics = {"accuracy": 0.95, "f1": 0.20}
    risk = performance.performance_risk(metrics)
    assert risk > 0.5


def test_performance_risk_is_zero_above_both_floors():
    metrics = {"accuracy": 0.90, "f1": 0.85}
    assert performance.performance_risk(metrics) == 0.0
