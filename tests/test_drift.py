"""Tests for fairguard.drift (PSI + KS)."""
import numpy as np
import pandas as pd

from fairguard import drift


def test_psi_identical_is_zero():
    values = pd.Series(np.linspace(0, 100, 300))
    assert drift.psi(values, values) == 0.0


def test_psi_flags_shifted_distribution():
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(50, 10, 500))
    current = pd.Series(rng.normal(80, 10, 500))  # clearly shifted
    value = drift.psi(reference, current)
    assert value > 0.25
    assert drift.psi_level(value) == "major"


def test_psi_level_boundaries():
    assert drift.psi_level(0.05) == "stable"
    assert drift.psi_level(0.15) == "moderate"
    assert drift.psi_level(0.5) == "major"


def test_categorical_psi():
    values = pd.Series(["a", "b", "a", "b"])
    assert drift.categorical_psi(values, values) == 0.0
    shifted = pd.Series(["a", "a", "a", "b"])
    assert drift.categorical_psi(values, shifted) > 0.0


def test_ks_identical_samples():
    values = pd.Series(np.linspace(0, 1, 250))
    stat, pvalue = drift.ks(values, values)
    assert stat == 0.0
    assert pvalue > 0.05


def test_assess_drift_report(demo_df, drifted_df):
    result = drift.assess_drift(demo_df, drifted_df)
    assert "applicant_id" not in result  # identifiers excluded
    assert "income_lakhs" in result
    entry = result["income_lakhs"]
    assert set(entry) == {"psi", "psi_level", "ks_stat", "ks_pvalue", "missing_pct_shift"}
    assert entry["psi_level"] in {"moderate", "major"}
    assert result["gender"]["ks_stat"] is None  # categorical -> no KS


def test_drift_risk():
    assert drift.drift_risk({}) == 0.0
    stable = {"x": {"psi_level": "stable"}, "y": {"psi_level": "stable"}}
    assert drift.drift_risk(stable) == 0.0
    # RC-06: the worst feature always keeps half of the score, so a single
    # major shift among two features is 0.5 * mean + 0.5 * peak = 0.75
    # instead of the old flat mean of 0.5.
    mixed = {"x": {"psi_level": "major"}, "y": {"psi_level": "stable"}}
    assert drift.drift_risk(mixed) == 0.75


def test_drift_risk_is_importance_aware():
    """A shift in a feature the model relies on must score worse (RC-06)."""
    result = {
        "a": {"psi_level": "major"},
        "b": {"psi_level": "moderate"},
    }
    a_matters = drift.drift_risk(result, importance={"a": 10.0, "b": 0.01})
    b_matters = drift.drift_risk(result, importance={"a": 0.01, "b": 10.0})
    assert a_matters > b_matters


def test_drift_risk_single_critical_feature_is_not_diluted():
    """One major shift among twenty features must not read as 'no risk' (RC-06)."""
    result = {f"f{i}": {"psi_level": "stable"} for i in range(19)}
    result["income_lakhs"] = {"psi_level": "major"}
    assert drift.drift_risk(result) >= 0.5


def test_aggregate_importance_maps_encoded_back_to_raw():
    folded = drift.aggregate_importance(
        {"gender_female": 0.3, "gender_male": 0.2, "credit_score": 0.5})
    assert folded["gender"] == 0.5
    assert folded["credit_score"] == 0.5


def test_aggregate_importance_uses_raw_feature_list():
    folded = drift.aggregate_importance(
        {"gender_female": 0.3, "gender_male": 0.2, "credit_score": 0.5},
        raw_features=["gender", "credit_score"])
    assert folded["gender"] == 0.5
    assert folded["credit_score"] == 0.5
    assert "credit" not in folded
