"""Tests for the fairness confidence intervals and small-group handling.

Covers RC-18 (no statistical uncertainty on fairness numbers) and RC-11
(crashes on NaN sensitive values).
"""
import numpy as np
import pandas as pd

from fairguard import config, fairness


def _biased_data(n_per_group=200):
    """Two groups with a real selection-rate gap and enough rows to bootstrap."""
    sensitive = pd.Series(["A"] * n_per_group + ["B"] * n_per_group)
    y_true = pd.Series([1] * (2 * n_per_group))
    # Group A: 80 % approved.  Group B: 50 % approved.
    approved_a = int(round(n_per_group * 0.8))
    approved_b = n_per_group // 2
    y_pred = pd.Series(
        [1] * approved_a + [0] * (n_per_group - approved_a)
        + [1] * approved_b + [0] * (n_per_group - approved_b)
    )
    assert len(y_true) == len(y_pred) == len(sensitive)
    return y_true, y_pred, sensitive


def test_assess_fairness_reports_confidence_intervals():
    y_true, y_pred, sensitive = _biased_data()
    result = fairness.assess_fairness(y_true, y_pred, sensitive, "group")
    assert "demographic_parity_difference_ci" in result
    assert "equalized_odds_difference_ci" in result
    low, high = result["demographic_parity_difference_ci"]
    assert low <= high
    assert 0.0 <= low <= 1.0 and 0.0 <= high <= 1.0
    assert result["n_boot"] == config.FAIRNESS_BOOTSTRAP
    assert result["ci_level"] == 95


def test_confidence_interval_brackets_the_point_estimate():
    """The interval must not contradict the point estimate it belongs to."""
    y_true, y_pred, sensitive = _biased_data()
    result = fairness.assess_fairness(y_true, y_pred, sensitive)
    point = result["demographic_parity_difference"]
    low, high = result["demographic_parity_difference_ci"]
    assert low <= point <= high


def test_no_intervals_for_small_samples():
    """A 10-row split is noise; the module must refuse to pretend otherwise."""
    y_true, y_pred, sensitive = _biased_data(n_per_group=5)
    result = fairness.assess_fairness(y_true, y_pred, sensitive)
    assert "demographic_parity_difference_ci" not in result
    assert result["small_groups"] == ["A", "B"]


def test_small_groups_are_flagged():
    y_true, y_pred, sensitive = _biased_data(n_per_group=10)
    result = fairness.assess_fairness(y_true, y_pred, sensitive)
    assert result["small_groups"] == ["A", "B"]


def test_bootstrap_is_reproducible():
    y_true, y_pred, sensitive = _biased_data()
    first = fairness.assess_fairness(y_true, y_pred, sensitive)
    second = fairness.assess_fairness(y_true, y_pred, sensitive)
    assert first["demographic_parity_difference_ci"] == second["demographic_parity_difference_ci"]


def test_missing_sensitive_values_are_dropped_not_fatal():
    """RC-11: MetricFrame used to raise on NaN groups."""
    y_true, y_pred, sensitive = _biased_data(n_per_group=60)
    sensitive = sensitive.astype(object)
    sensitive.iloc[0:5] = None
    result = fairness.assess_fairness(y_true, y_pred, sensitive)
    assert result["dropped_missing_groups"] == 5
    assert result["group_sizes"]["A"] == 55


def test_effective_disparity_uses_the_interval_midpoint_when_unclear():
    """An interval that straddles the limit is scored from its middle."""
    # limit 0.1 -> the point estimate 0.1 sits exactly on the boundary
    assert fairness._effective_disparity(0.10, [0.02, 0.05], 0.10) == 0.10
    # interval entirely above the limit -> the measured size is used
    assert fairness._effective_disparity(0.30, [0.20, 0.40], 0.10) == 0.30
    # interval straddling the limit -> midpoint
    assert fairness._effective_disparity(0.10, [0.05, 0.25], 0.10) == 0.15


def test_fairness_risk_uses_intervals_when_available():
    """A disparity whose interval crosses the limit must not be waved away."""
    base = {
        "demographic_parity_difference": 0.10,
        "equalized_odds_difference": 0.0,
    }
    confident = fairness.fairness_risk(dict(base))
    uncertain = fairness.fairness_risk(
        dict(base, demographic_parity_difference_ci=[0.05, 0.25])
    )
    assert uncertain > confident


def test_equalized_odds_limit_is_tighter_than_demographic_parity():
    """RC-04: the recommendation logic used to look at DPD only."""
    result = {
        "demographic_parity_difference": 0.01,
        "equalized_odds_difference": 0.30,
    }
    risk = fairness.fairness_risk(result)
    # 0.30 / 0.10 limit = 3x the limit -> capped at 1.0
    assert risk == 1.0
