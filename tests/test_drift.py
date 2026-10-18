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
    mixed = {"x": {"psi_level": "major"}, "y": {"psi_level": "stable"}}
    assert drift.drift_risk(mixed) == 0.5
