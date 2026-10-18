"""Risk scoring - combine per-dimension risks into one score and plain-language advice."""
from . import config


def risk_level(score: float) -> str:
    """Low / Medium / High / Critical for a 0..1 score (bounds in config)."""
    for bound, name in config.RISK_LEVEL_BOUNDS:
        if score <= bound:
            return name
    return "Critical"


def compute_risk(risks: dict) -> dict:
    """Weighted risk over the dimensions that were actually assessed.

    risks: {"fairness": 0.7, "privacy": 0.4, ...} with values in 0..1.
    Returns score (0..1), level, and each dimension's contribution to the score.
    """
    weights = {k: config.RISK_WEIGHTS[k] for k in risks if k in config.RISK_WEIGHTS}
    if not weights:
        return {"score": 0.0, "level": "Low", "contributions": {}}
    total = sum(weights.values())
    score = sum(risks[k] * w for k, w in weights.items()) / total
    return {
        "score": round(score, 4),
        "level": risk_level(score),
        "contributions": {k: round(risks[k] * w / total, 4) for k, w in weights.items()},
    }


def deployment_decision(level: str) -> str:
    """Suggested deployment action for a risk level (the approval workflow)."""
    return {
        "Low": "Approve",
        "Medium": "Approve with monitoring",
        "High": "Needs review",
        "Critical": "Block",
    }[level]


def build_recommendations(assessment: dict) -> list:
    """Plain-language actions based on whatever the assessment actually found."""
    recommendations = []

    fairness = assessment.get("fairness") or {}
    if fairness.get("demographic_parity_difference", 0) > config.FAIRNESS_DPD_LIMIT:
        recommendations.append(
            "Fairness: approval rates differ across groups - check features that act as "
            "proxies for the sensitive attribute and consider mitigation such as Fairlearn's ThresholdOptimizer."
        )

    performance = assessment.get("performance") or {}
    if performance.get("accuracy", 1) < config.MIN_ACCURACY:
        recommendations.append(
            "Performance: accuracy is below the 0.70 floor - retrain or fix data quality before deploying."
        )

    privacy = assessment.get("privacy") or {}
    if privacy.get("findings"):
        recommendations.append(
            "Privacy: PII-like columns found - drop or mask them before training, keep only aggregated identifiers."
        )

    drift = assessment.get("drift") or {}
    major = [column for column, entry in drift.items() if entry.get("psi_level") == "major"]
    if major:
        recommendations.append(
            f"Drift: major distribution shift in {', '.join(major)} - re-validate on recent data and schedule retraining."
        )

    text = assessment.get("text") or {}
    if text.get("flagged", 0) > 0:
        recommendations.append(
            "Text safety: some generated outputs were flagged - add filtering or regeneration guardrails."
        )

    if not recommendations:
        recommendations.append("No blocking issues found - keep monitoring with scheduled assessments.")
    return recommendations
