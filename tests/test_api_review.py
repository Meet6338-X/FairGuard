"""Tests for the FastAPI layer — RC-16 and RC-07.

``POST /api/review`` used to ignore the ``run_id`` in the request and always
overwrite the **newest** assessment on disk, so a reviewer's decision could be
attached to the wrong run.  ``/api/assess`` also defaulted to comparing the
dataset with itself, which made every PSI exactly 0.
"""
import json

import pytest
from fastapi.testclient import TestClient

import api
from fairguard import config, report


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


@pytest.fixture
def two_reports(tmp_path, monkeypatch):
    """Two saved assessments with different ids."""
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path)
    base = {
        "model_id": "demo_model",
        "model_version": "1.0",
        "dataset_id": "demo_data",
        "sensitive_attribute": "gender",
        "threshold": 0.5,
        "assessed_at": "2026-10-17T10:30:00+00:00",
        "performance": {"accuracy": 0.8, "precision": 0.7, "recall": 0.6,
                        "f1": 0.65, "roc_auc": 0.85,
                        "confusion_matrix": [[60, 15], [22, 43]]},
        "fairness": {"demographic_parity_difference": 0.2,
                     "equalized_odds_difference": 0.1,
                     "by_group": {}, "group_sizes": {}},
        "privacy": {"findings": [], "risk": 0.0},
        "explainability": {"global_importance": {"age": 0.2}},
        "risk": {"score": 0.3, "level": "Medium", "contributions": {},
                 "dimensions_assessed": ["performance"], "coverage": 1.0},
        "recommendations": [],
        "decision": "Needs review",
        "reviewer_decision": None,
    }
    first = dict(base, assessed_at="2026-10-17T10:30:00+00:00")
    second = dict(base, assessed_at="2026-10-17T11:45:12+00:00")
    path_one = tmp_path / "assessment_20261017T103000.json"
    path_two = tmp_path / "assessment_20261017T114512.json"
    path_one.write_text(json.dumps(first), encoding="utf-8")
    path_two.write_text(json.dumps(second), encoding="utf-8")
    return {"one": path_one, "two": path_two}


def test_review_writes_to_the_requested_run_only(client, two_reports):
    """RC-16: the decision must land on the run the reviewer named."""
    response = client.post("/api/review", json={
        "run_id": "20261017T103000",
        "decision": "Block",
        "note": "Fairness gap too wide.",
    })
    assert response.status_code == 200

    one = json.loads(two_reports["one"].read_text(encoding="utf-8"))
    two = json.loads(two_reports["two"].read_text(encoding="utf-8"))
    assert one["reviewer_decision"] == "Block"
    assert one["reviewer_note"] == "Fairness gap too wide."
    assert one.get("reviewed_at")
    # the other assessment must be untouched
    assert two["reviewer_decision"] is None


def test_review_on_an_unknown_run_is_a_404(client, two_reports):
    response = client.post("/api/review", json={
        "run_id": "does-not-exist",
        "decision": "Block",
    })
    assert response.status_code == 404


def test_review_rejects_an_invalid_decision(client, two_reports):
    response = client.post("/api/review", json={
        "run_id": "20261017T103000",
        "decision": "Probably fine",
    })
    assert response.status_code == 400


def test_report_export_finds_the_right_run(client, two_reports):
    response = client.get("/api/report/20261017T103000/md")
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "assessment_20261017T103000.md"
    assert "FairGuard AI" in body["content"]


def test_report_export_of_an_unknown_run_is_a_404(client, two_reports):
    assert client.get("/api/report/nope/json").status_code == 404


def test_assess_request_does_not_default_to_a_self_reference():
    """RC-07: `use_reference=True` made every PSI exactly 0."""
    assert api.AssessRequest(dataset="d.csv", model="m.pkl").use_reference is False
    assert api.AssessRequest(dataset="d.csv", model="m.pkl").reference is None


def test_collision_suffixed_reports_are_found(client, two_reports):
    """Two assessments in the same second must not overwrite each other."""
    original = two_reports["one"].read_text(encoding="utf-8")
    suffixed = two_reports["one"].with_name("assessment_20261017T103000_2.json")
    suffixed.write_text(original, encoding="utf-8")
    response = client.get("/api/report/20261017T103000/json")
    assert response.status_code == 200
