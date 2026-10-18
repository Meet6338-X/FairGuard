"""Tests for fairguard.tracking (MLflow, local file store)."""
from fairguard import tracking


def test_log_assessment_and_history(tmp_path, sample_assessment):
    run_id = tracking.log_assessment(sample_assessment, tracking_dir=tmp_path)
    assert isinstance(run_id, str) and run_id

    frame = tracking.history(tracking_dir=tmp_path)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["model"] == "demo_model"
    assert round(float(row["risk"]), 2) == 0.42
    assert row["level"] == "Medium"
    assert row["decision"] == "Approve with monitoring"
    assert row["reviewer"] == "pending"


def test_history_without_runs_is_empty(tmp_path):
    frame = tracking.history(tracking_dir=tmp_path / "fresh")
    assert frame.empty
