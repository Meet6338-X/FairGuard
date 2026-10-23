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


def _ci(entry: dict, key: str) -> str:
    """Render a [low, high] interval as '0.12 [0.08 – 0.17]'."""
    interval = entry.get(key)
    if not interval:
        return "—"
    return f"{entry.get(key.replace('_ci', ''))} [{interval[0]} – {interval[1]}]"


def build_markdown(assessment: dict) -> str:
    """Human-readable, audit-ready assessment report."""
    evaluation = assessment.get("evaluation") or {}
    risk = assessment.get("risk") or {}
    lines = [
        "# FairGuard AI — Assessment Report",
        "",
        f"**Model:** {assessment['model_id']} (version {assessment['model_version']})  ",
        f"**Dataset:** {assessment['dataset_id']}  ",
        f"**Sensitive attribute:** {assessment['sensitive_attribute']}  ",
        f"**Assessed at:** {assessment['assessed_at']}",
        "",
    ]

    # RC-02: the reader must know whether the numbers are out-of-sample.
    if evaluation:
        scope = evaluation.get("scope", "full_dataset")
        scope_note = (
            "Hold-out split — the model was **not** trained on these rows."
            if scope == "holdout" else
            "Full dataset — the model was trained on part of these rows, so these "
            "numbers are **in-sample** and optimistic."
        )
        lines += [
            "## Evaluation scope",
            "",
            f"- Scope: **{scope}** ({scope_note})",
            f"- Rows scored: {evaluation.get('n_rows')} of {evaluation.get('n_dataset_rows')}",
            "",
        ]

    # RC-23: a detected label duplicate invalidates every number below.
    leakage = assessment.get("leakage") or {}
    if leakage.get("columns"):
        lines += [
            "## ⚠️ Label leakage detected",
            "",
            "Feature column(s) " + ", ".join(f"`{c}`" for c in leakage["columns"]) +
            " map **1:1 onto the label** (" + leakage.get("note", "") + "). "
            "Every metric in this report is unreliable until the model is retrained "
            "without them.",
            "",
        ]

    lines += [
        "## Risk summary",
        "",
        f"**Risk score {risk.get('score', 0):.2f} → {risk.get('level', '—')}** — "
        f"suggested decision: **{assessment.get('decision', '—')}**",
        "",
    ]
    dimensions = risk.get("dimensions_assessed") or []
    if dimensions:
        lines += [
            f"- Dimensions assessed: {', '.join(dimensions)}",
            f"- Risk-weight coverage: {risk.get('coverage', 0):.0%} of the total weight — "
            "scores are only comparable between assessments with the same coverage.",
            "",
        ]
    lines += [
        "| Dimension | Contribution |",
        "| --- | --- |",
    ]
    for dimension, contribution in (risk.get("contributions") or {}).items():
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
    if perf.get("brier_score") is not None:
        lines += [
            f"- Brier score: {_metric(perf.get('brier_score'))}",
            f"- Expected calibration error: {_metric(perf.get('calibration_error'))}",
        ]

    fairness = assessment.get("fairness") or {}
    if fairness:
        lines += [
            "",
            "## Fairness",
            "",
            f"- Demographic parity difference: {_ci(fairness, 'demographic_parity_difference_ci')}",
            f"- Equalized odds difference: {_ci(fairness, 'equalized_odds_difference_ci')}",
        ]
        if fairness.get("small_groups"):
            lines.append(
                "- Groups with fewer than "
                f"{config.MIN_GROUP_SIZE} rows (metrics are noisy): "
                + ", ".join(str(g) for g in fairness["small_groups"])
            )
        lines += [
            "",
            "| Group | Rows | Selection rate | Accuracy | Precision | Recall |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        groups = fairness.get("by_group", {})
        sizes = fairness.get("group_sizes", {})
        for group in groups.get("selection_rate", {}):
            values = [str(sizes.get(group, "—"))]
            values += [str(groups.get(metric, {}).get(group, "—"))
                       for metric in ("selection_rate", "accuracy", "precision", "recall")]
            lines.append("| " + " | ".join([str(group), *values]) + " |")

    explainability = assessment.get("explainability") or {}
    importance = explainability.get("global_importance") or {}
    if importance:
        lines += [
            "",
            "## Explainability — top features (mean |SHAP|)",
            "",
            f"- Explainer: `{explainability.get('explainer', 'unknown')}` "
            f"(background {explainability.get('n_background', '—')} rows)",
            "",
        ]
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
        reference = assessment.get("drift_reference") or {}
        if reference.get("same_as_assessed"):
            lines += [
                "",
                "> ⚠️ The reference dataset is the assessed dataset itself, so every PSI is 0 "
                "by construction. This is a baseline, **not** evidence that the data is stable.",
            ]
        elif reference:
            lines += ["", f"> Drift baseline: a separate reference dataset ({reference.get('n_rows', '—')} rows)."]

    text = assessment.get("text") or {}
    if text:
        lines += [
            "",
            "## Text / LLM safety",
            "",
            f"- Backend: {text.get('backend')}",
            f"- Overall risk: {text.get('risk')} · flagged {text.get('flagged')} / {text.get('n_texts')} texts",
        ]
        backends = text.get("backends") or {}
        if backends:
            lines.append("- Backend per check: " + ", ".join(f"{k}={v}" for k, v in backends.items()))
        summary = text.get("kind_summary") or {}
        if summary:
            lines.append("- Mean score per check: " + ", ".join(f"{k}={v}" for k, v in summary.items()))

    lines += ["", "## Recommendations", ""]
    lines += [f"- {rec}" for rec in assessment.get("recommendations", [])]

    lines += [
        "",
        "## Reviewer decision",
        "",
        f"- Suggested: **{assessment.get('decision', '—')}**",
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
    evaluation = assessment.get("evaluation") or {}
    leakage = assessment.get("leakage") or {}
    importance = (assessment.get("explainability") or {}).get("global_importance") or {}
    top = ", ".join(list(importance)[:5]) or "—"
    scope = evaluation.get("scope", "full_dataset")
    scope_text = ("hold-out (out-of-sample)" if scope == "holdout"
                  else "full dataset (in-sample — optimistic)")
    return "\n".join([
        f"# Model Card — {assessment['model_id']}",
        "",
        "## Model details",
        f"- Version: {assessment['model_version']}",
        f"- Assessed on: {assessment['dataset_id']} ({assessment['assessed_at']})",
        f"- Evaluation scope: {scope_text} · {evaluation.get('n_rows', '—')} rows scored",
        *(
            [f"- **⚠️ Label leakage detected**: columns {', '.join(leakage['columns'])} "
             "are 1:1 copies of the label — the metrics are unreliable"]
            if leakage.get("columns") else []
        ),
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
        f"- Calibration error (ECE) {_metric(perf.get('calibration_error'))} · Brier {_metric(perf.get('brier_score'))}",
        f"- Demographic parity difference: {fairness.get('demographic_parity_difference', '—')}",
        f"- Equalized odds difference: {fairness.get('equalized_odds_difference', '—')}",
        "",
        "## Key drivers (mean |SHAP|)",
        f"- {top}",
        "",
        "## Risk assessment",
        f"- FairGuard risk score {assessment['risk']['score']:.2f} → {assessment['risk']['level']}",
        f"- Dimensions assessed: {', '.join(assessment['risk'].get('dimensions_assessed') or []) or '—'}",
        f"- Suggested decision: {assessment['decision']}",
        "",
        "## Caveats and recommendations",
        f"- {DISCLAIMER}",
        *[f"- {rec}" for rec in assessment.get("recommendations", [])],
        "",
        MODEL_CARD_NOTE,
    ])


def save_report(assessment: dict, out_dir=None) -> dict:
    """Write JSON + Markdown report + model card to disk; returns the paths.

    Two assessments started in the same second used to overwrite each other
    because the filename only carried second-level precision.  A numeric
    suffix is added when a file with that name already exists.
    """
    out = Path(out_dir or config.REPORTS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    stamp = assessment["assessed_at"].replace(":", "").replace("-", "")[:15]
    paths = {
        "json": out / f"assessment_{stamp}.json",
        "markdown": out / f"assessment_{stamp}.md",
        "model_card": out / f"model_card_{stamp}.md",
    }
    if paths["json"].exists():
        counter = 2
        while paths["json"].exists():
            paths = {
                "json": out / f"assessment_{stamp}_{counter}.json",
                "markdown": out / f"assessment_{stamp}_{counter}.md",
                "model_card": out / f"model_card_{stamp}_{counter}.md",
            }
            counter += 1
    paths["json"].write_text(to_json(assessment), encoding="utf-8")
    paths["markdown"].write_text(build_markdown(assessment), encoding="utf-8")
    paths["model_card"].write_text(build_model_card(assessment), encoding="utf-8")
    return {kind: str(path) for kind, path in paths.items()}
