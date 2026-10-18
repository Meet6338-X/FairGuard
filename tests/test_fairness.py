"""Tests for fairguard.fairness."""
import pandas as pd

from fairguard import fairness


def _toy_data():
    # 50 people in group A are approved, 50 in group B are all rejected.
    sensitive = pd.Series(["A"] * 50 + ["B"] * 50)
    y_pred = pd.Series([1] * 50 + [0] * 50)
    y_true = pd.Series([1] * 50 + [1] * 50)  # everyone deserved approval
    return y_true, y_pred, sensitive


def test_assess_fairness_structure():
    y_true, y_pred, sensitive = _toy_data()
    result = fairness.assess_fairness(y_true, y_pred, sensitive, "group")
    assert result["sensitive_attribute"] == "group"
    assert set(result["by_group"]) == {"selection_rate", "accuracy", "precision", "recall"}
    assert result["group_sizes"] == {"A": 50, "B": 50}
    assert result["demographic_parity_difference"] == 1.0
    assert 0 <= result["equalized_odds_difference"] <= 1


def test_fairness_risk_scales_with_gap():
    y_true, y_pred, sensitive = _toy_data()
    result = fairness.assess_fairness(y_true, y_pred, sensitive)
    assert fairness.fairness_risk(result) == 1.0  # huge gap -> max risk

    mild = dict(result)
    mild["demographic_parity_difference"] = 0.10  # exactly at the limit
    mild["equalized_odds_difference"] = 0.0
    assert fairness.fairness_risk(mild) == 0.5


def test_fairness_no_gap_is_low_risk():
    y_true = pd.Series([1, 0, 0, 1])
    y_pred = pd.Series([1, 0, 0, 1])
    sensitive = pd.Series(["A", "B", "A", "B"])  # both groups 50% approved
    result = fairness.assess_fairness(y_true, y_pred, sensitive)
    assert result["demographic_parity_difference"] == 0.0
    assert fairness.fairness_risk(result) == 0.0
