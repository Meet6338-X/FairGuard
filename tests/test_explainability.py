"""Tests for fairguard.explainability (SHAP)."""
from fairguard import explainability


def test_global_importance_covers_all_features(bundle):
    importance = explainability.global_importance(bundle["model"], bundle["X_test"])
    assert set(importance) == set(bundle["X_test"].columns)
    values = list(importance.values())
    assert values == sorted(values, reverse=True)  # largest first
    assert all(v >= 0 for v in values)


def test_local_explanation_single_row(bundle):
    result = explainability.local_explanation(bundle["model"], bundle["X_test"], row_index=3)
    assert result["row_index"] == 3
    assert 0 <= result["probability"] <= 1
    assert set(result["contributions"]) == set(bundle["X_test"].columns)
