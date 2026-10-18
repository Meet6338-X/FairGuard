"""Tests for fairguard.performance."""
from fairguard import performance


def test_evaluate_performance_perfect():
    metrics = performance.evaluate_performance(
        [0, 1, 0, 1], [0, 1, 0, 1], y_prob=[0.1, 0.9, 0.2, 0.8]
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_evaluate_performance_without_probabilities():
    metrics = performance.evaluate_performance([0, 1, 0, 1], [1, 1, 0, 0])
    assert "roc_auc" not in metrics
    assert 0 <= metrics["f1"] <= 1


def test_performance_risk():
    assert performance.performance_risk({"accuracy": 0.9}) == 0.0
    assert performance.performance_risk({"accuracy": 0.35}) == 0.5
    assert performance.performance_risk({"accuracy": 0.0}) == 1.0
