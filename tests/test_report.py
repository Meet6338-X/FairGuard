"""Tests for fairguard.report — JSON, Markdown, model card."""
import json

from fairguard import report


def test_to_json_roundtrip(sample_assessment):
    loaded = json.loads(report.to_json(sample_assessment))
    assert loaded["model_id"] == "demo_model"
    assert loaded["risk"]["level"] == "Medium"


def test_build_markdown_sections(sample_assessment):
    markdown = report.build_markdown(sample_assessment)
    for section in ("# FairGuard AI", "## Risk summary", "## Performance",
                    "## Fairness", "## Explainability", "## Privacy findings",
                    "## Drift", "## Recommendations", "## Reviewer decision"):
        assert section in markdown
    assert "statistical measurements" in markdown  # the disclaimer


def test_build_model_card(sample_assessment):
    card = report.build_model_card(sample_assessment)
    assert "Model Card — demo_model" in card
    assert "## Model details" in card
    assert "## Intended use" in card
    assert "Not validated" in card
    assert "credit_score" in card  # top key driver from the sample


def test_save_report_writes_three_files(tmp_path, sample_assessment):
    paths = report.save_report(sample_assessment, out_dir=tmp_path)
    assert set(paths) == {"json", "markdown", "model_card"}
    for path in paths.values():
        from pathlib import Path
        assert Path(path).exists()
    json.loads(open(paths["json"], encoding="utf-8").read())
