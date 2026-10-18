"""Assessment report and automated model card — JSON + Markdown exports."""
import json
from pathlib import Path

from . import config

DISCLAIMER = (
    "Fairness results are statistical measurements of *this model on this dataset*. "
    "They are not conclusions about protected groups of people."
)
MODEL_CARD_NOTE = (
    "Generated automatically from a FairGuard assessment, in the spirit of "
    "Mitchell et al., 'Model Cards for Model Reporting' (2019)."
)


def to_json(assessment: dict) -> str:
    """Assessment record as pretty JSON (SRS schema, plus a few extras)."""
    return json.dumps(assessment, indent=2, default=str)


def _metric(value, digits=3) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "—"


def build_markdown(assessment: dict) -> str:
    """Human-readable, audit-ready assessment report."""
    lines = [
        "# FairGuard AI — Assessment Report",
        "",
        f"**Model:** {assessment['model_id']} (version {assessment['model_version']})  ",
        f"**Dataset:** {assessment['dataset_id']}  ",
        f"**Sensitive attribute:** {assessment['sensitive_attribute']}  ",
        f"**Assessed at:** {assessment['assessed_at']}",
        "",
        "## Risk summary",
        "",
        f"**Risk score {assessment['risk']['score']:.2f} → {assessment['risk']['level']}** — "
        f"suggested decision: **{assessment['decision']}**",
        "",
        "| Dimension | Contribution |",
        "| --- | --- |",
    ]
    for dimension, contribution in assessment["risk"]["contributions"].items():
        lines.append(f"| {dimension} | {contribution:.3f} |")

    perf = assessment.get("performance") or {}
    lines += [
        "",
        "## Performance",
        "",
        f"- Accuracy: {_metric(perf.get('accuracy'))}",
        f"- Precision: {_metric(perf.get('precision'))}",
        f"- Recall: {_metric(perf.get('recall'))}",
        f"- F1: {_metric(perf.get('f1'))}",
        f"- ROC-AUC: {_metric(perf.get('roc_auc'))}",
    ]

    fairness = assessment.get("fairness") or {}
    if fairness:
        lines += [
            "",
            "## Fairness",
            "",
            f"- Demographic parity difference: {fairness.get('demographic_parity_difference')}",
            f"- Equalized odds difference: {fairness.get('equalized_odds_difference')}",
            "",
            "| Group | Selection rate | Accuracy | Precision | Recall |",
            "| --- | --- | --- | --- | --- |",
        ]
        groups = fairness.get("by_group", {})
        for group in groups.get("selection_rate", {}):
            values = [str(groups.get(metric, {}).get(group, "—"))
                      for metric in ("selection_rate", "accuracy", "precision", "recall")]
            lines.append("| " + " | ".join([str(group), *values]) + " |")

    importance = (assessment.get("explainability") or {}).get("global_importance") or {}
    if importance:
        lines += ["", "## Explainability — top features (mean |SHAP|)", ""]
        for feature, value in list(importance.items())[:10]:
            lines.append(f"- {feature}: {value}")

    privacy = assessment.get("privacy") or {}
    if privacy.get("findings"):
        lines += [
            "",
            "## Privacy findings",
            "",
            "| Column | Type | Rows affected | Severity |",
            "| --- | --- | --- | --- |",
        ]
        for finding in privacy["findings"]:
            lines.append(
                f"| {finding['column']} | {finding['type']} | {finding['matches']} | {finding['severity']} |"
            )

    drift = assessment.get("drift") or {}
    if drift:
        lines += ["", "## Drift (PSI / KS)", "", "| Feature | PSI | Level | KS p-value |", "| --- | --- | --- | --- |"]
        for feature, entry in drift.items():
            lines.append(f"| {feature} | {entry['psi']} | {entry['psi_level']} | {entry.get('ks_pvalue')} |")

    text = assessment.get("text") or {}
    if text:
        lines += [
            "",
            "## Text / LLM safety",
            "",
            f"- Backend: {text.get('backend')}",
            f"- Overall risk: {text.get('risk')} · flagged {text.get('flagged')} / {text.get('n_texts')} texts",
        ]
        summary = text.get("kind_summary") or {}
        if summary:
            lines.append("- Mean score per check: " + ", ".join(f"{k}={v}" for k, v in summary.items()))

    lines += ["", "## Recommendations", ""]
    lines += [f"- {rec}" for rec in assessment.get("recommendations", [])]

    lines += [
        "",
        "## Reviewer decision",
        "",
        f"- Suggested: **{assessment['decision']}**",
        f"- Reviewer: {assessment.get('reviewer_decision') or '_pending_'}",
        f"- Note: {assessment.get('reviewer_note') or '—'}",
        "",
        "---",
        "",
        DISCLAIMER,
        "",
    ]
    return "\n".join(lines)


def build_model_card(assessment: dict) -> str:
    """Automated model card for the assessed model."""
    perf = assessment.get("performance") or {}
    fairness = assessment.get("fairness") or {}
    importance = (assessment.get("explainability") or {}).get("global_importance") or {}
    top = ", ".join(list(importance)[:5]) or "—"
    return "\n".join([
        f"# Model Card — {assessment['model_id']}",
        "",
        "## Model details",
        f"- Version: {assessment['model_version']}",
        f"- Assessed on: {assessment['dataset_id']} ({assessment['assessed_at']})",
        "",
        "## Intended use",
        "- Academic demonstration of the FairGuard workbench.",
        "- **Not validated** for real credit, employment or other high-stakes decisions.",
        "",
        "## Factors",
        f"- Sensitive attribute assessed: {assessment['sensitive_attribute']}",
        f"- Decision threshold: {assessment.get('threshold', 0.5)}",
        "",
        "## Metrics",
        f"- Accuracy {_metric(perf.get('accuracy'))} · F1 {_metric(perf.get('f1'))} · ROC-AUC {_metric(perf.get('roc_auc'))}",
        f"- Demographic parity difference: {fairness.get('demographic_parity_difference', '—')}",
        f"- Equalized odds difference: {fairness.get('equalized_odds_difference', '—')}",
        "",
        "## Key drivers (mean |SHAP|)",
        f"- {top}",
        "",
        "## Risk assessment",
        f"- FairGuard risk score {assessment['risk']['score']:.2f} → {assessment['risk']['level']}",
        f"- Suggested decision: {assessment['decision']}",
        "",
        "## Caveats and recommendations",
        f"- {DISCLAIMER}",
        *[f"- {rec}" for rec in assessment.get("recommendations", [])],
        "",
        MODEL_CARD_NOTE,
    ])


def save_report(assessment: dict, out_dir=None) -> dict:
    """Write JSON + Markdown report + model card to disk; returns the paths."""
    out = Path(out_dir or config.REPORTS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    stamp = assessment["assessed_at"].replace(":", "").replace("-", "")[:15]
    paths = {
        "json": out / f"assessment_{stamp}.json",
        "markdown": out / f"assessment_{stamp}.md",
        "model_card": out / f"model_card_{stamp}.md",
    }
    paths["json"].write_text(to_json(assessment), encoding="utf-8")
    paths["markdown"].write_text(build_markdown(assessment), encoding="utf-8")
    paths["model_card"].write_text(build_model_card(assessment), encoding="utf-8")
    return {kind: str(path) for kind, path in paths.items()}
