"""Tests for fairguard.risk — scoring, levels, decisions, recommendations."""
from fairguard import risk


def test_risk_level_bounds():
    assert risk.risk_level(0.1) == "Low"
    assert risk.risk_level(0.4) == "Medium"
    assert risk.risk_level(0.6) == "High"
    assert risk.risk_level(0.9) == "Critical"


def test_compute_risk_single_dimension():
    result = risk.compute_risk({"fairness": 1.0})
    assert result["score"] == 1.0
    assert result["level"] == "Critical"
    assert result["contributions"] == {"fairness": 1.0}


def test_compute_risk_normalises_over_present_dimensions():
    result = risk.compute_risk({"fairness": 1.0, "performance": 0.0})
    # weights 0.30 + 0.15 => fairness contributes 2/3 of the score
    assert result["score"] == 0.6667
    assert result["level"] == "High"


def test_compute_risk_ignores_unknown_dimensions():
    result = risk.compute_risk({"fairness": 1.0, "astrology": 1.0})
    assert result["score"] == 1.0
    assert "astrology" not in result["contributions"]
    assert risk.compute_risk({})["score"] == 0.0


def test_deployment_decisions():
    assert risk.deployment_decision("Low") == "Approve"
    assert risk.deployment_decision("Medium") == "Approve with monitoring"
    assert risk.deployment_decision("High") == "Needs review"
    assert risk.deployment_decision("Critical") == "Block"


def test_build_recommendations_default():
    recommendations = risk.build_recommendations({})
    assert len(recommendations) == 1
    assert "monitoring" in recommendations[0]


def test_build_recommendations_per_dimension():
    recommendations = risk.build_recommendations({
        "fairness": {"demographic_parity_difference": 0.5},
        "privacy": {"findings": [{"column": "email"}]},
        "drift": {"income_lakhs": {"psi_level": "major"}},
        "text": {"flagged": 2},
    })
    joined = " ".join(recommendations)
    assert "Fairness:" in joined
    assert "Privacy:" in joined
    assert "Drift:" in joined
    assert "Text safety:" in joined
