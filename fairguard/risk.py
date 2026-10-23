"""Risk scoring - combine per-dimension risks into one score and plain-language advice.

Root causes fixed here
-----------------------
* **RC-04** the recommendations only looked at the demographic-parity
  difference.  A model with DPD 0.01 and EOD 0.30 — a serious equalized-odds
  violation — was told "No blocking issues found".  Both disparities are
  checked now.
* **RC-05** the risk score is normalised over the dimensions that were
  actually assessed, so the *same* underlying finding produced 0.18 "Low"
  with five dimensions and 0.2769 "Medium" with three.  The score now
  carries its coverage so two assessments can be compared honestly.
"""
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
    Returns score (0..1), level, each dimension's contribution to the score,
    the dimensions that were assessed and how much of the total risk weight
    they cover (RC-05).
    """
    weights = {k: config.RISK_WEIGHTS[k] for k in risks if k in config.RISK_WEIGHTS}
    if not weights:
        return {"score": 0.0, "level": "Low", "contributions": {},
                "dimensions_assessed": [], "coverage": 0.0}
    total = sum(weights.values())
    score = sum(risks[k] * w for k, w in weights.items()) / total
    coverage = total / sum(config.RISK_WEIGHTS.values())
    return {
        "score": round(score, 4),
        "level": risk_level(score),
        "contributions": {k: round(risks[k] * w / total, 4) for k, w in weights.items()},
        "dimensions_assessed": sorted(weights),
        "coverage": round(coverage, 4),
        "weights_used": {k: weights[k] for k in sorted(weights)},
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

    # RC-23 hardening: a model artifact trained before the leak guard may
    # still carry features that are copies of the label.  That finding
    # invalidates every other number on the report, so it goes first.
    leakage = assessment.get("leakage") or {}
    leak_columns = leakage.get("columns") or []
    if leak_columns:
        recommendations.append(
            "Leakage: the model uses " + ", ".join(leak_columns) +
            " which are 1:1 copies of the label - the metrics in this report are "
            "unreliable; retrain with these columns excluded."
        )

    fairness = assessment.get("fairness") or {}
    dpd = fairness.get("demographic_parity_difference", 0) or 0
    eod = fairness.get("equalized_odds_difference", 0) or 0
    # RC-04: equalized odds matters just as much as demographic parity.
    if dpd > config.FAIRNESS_DPD_LIMIT or eod > config.FAIRNESS_EOD_LIMIT:
        parts = []
        if dpd > config.FAIRNESS_DPD_LIMIT:
            parts.append(f"demographic parity difference {dpd:.2f}")
        if eod > config.FAIRNESS_EOD_LIMIT:
            parts.append(f"equalized odds difference {eod:.2f}")
        recommendations.append(
            "Fairness: " + " and ".join(parts) + " exceed the configured limits "
            f"(DPD {config.FAIRNESS_DPD_LIMIT:.2f} / EOD {config.FAIRNESS_EOD_LIMIT:.2f}) - "
            "check features that act as proxies for the sensitive attribute and consider "
            "mitigation such as Fairlearn's ThresholdOptimizer."
        )
    small_groups = fairness.get("small_groups") or []
    if small_groups:
        recommendations.append(
            "Fairness: group(s) " + ", ".join(str(g) for g in small_groups) +
            f" have fewer than {config.MIN_GROUP_SIZE} rows - their metrics are noisy, "
            "collect more data for them before drawing conclusions."
        )

    performance = assessment.get("performance") or {}
    accuracy = performance.get("accuracy", 1)
    f1 = performance.get("f1", 1)
    if accuracy < config.MIN_ACCURACY:
        recommendations.append(
            f"Performance: accuracy {accuracy:.2f} is below the {config.MIN_ACCURACY:.2f} floor - "
            "retrain or fix data quality before deploying."
        )
    elif f1 < config.MIN_F1:
        recommendations.append(
            f"Performance: F1 {f1:.2f} is below the {config.MIN_F1:.2f} floor - the model is "
            "missing the minority class, try class weighting or resampling."
        )
    calibration = performance.get("calibration_error")
    if calibration is not None and calibration > 0.15:
        recommendations.append(
            f"Performance: probabilities are poorly calibrated (expected calibration error "
            f"{calibration:.2f}) - apply isotonic/Platt calibration before using them for "
            "threshold decisions."
        )

    privacy = assessment.get("privacy") or {}
    if privacy.get("findings"):
        columns = sorted({f["column"] for f in privacy["findings"]})
        recommendations.append(
            "Privacy: PII-like columns found (" + ", ".join(columns[:8]) +
            (", ..." if len(columns) > 8 else "") +
            ") - drop or mask them before training, keep only aggregated identifiers."
        )

    drift = assessment.get("drift") or {}
    major = [column for column, entry in drift.items() if entry.get("psi_level") == "major"]
    moderate = [column for column, entry in drift.items() if entry.get("psi_level") == "moderate"]
    if major:
        recommendations.append(
            f"Drift: major distribution shift in {', '.join(major)} - re-validate on recent data and schedule retraining."
        )
    if moderate:
        recommendations.append(
            f"Drift: moderate distribution shift in {', '.join(moderate)} - watch these features and "
            "re-check after the next data drop."
        )

    text = assessment.get("text") or {}
    if text.get("flagged", 0) > 0:
        recommendations.append(
            "Text safety: some generated outputs were flagged - add filtering or regeneration guardrails."
        )

    if not recommendations:
        recommendations.append("No blocking issues found - keep monitoring with scheduled assessments.")
    return recommendations
