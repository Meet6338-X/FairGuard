"""Tests for the SHAP explainer selection — RC-01.

Every FairGuard model is an sklearn ``Pipeline``.  The old code checked
``isinstance(model, LogisticRegression)`` / ``isinstance(model,
RandomForestClassifier)``, which never matched, so *every* model silently fell
back to the slow model-agnostic ``PermutationExplainer``.
"""
import time

from fairguard import explainability, train


def test_explainer_kind_matches_the_estimator(demo_df):
    """The kind must follow the final estimator inside the pipeline."""
    for model_type, expected in (
        ("logistic_regression", "linear"),
        ("random_forest", "tree"),
        ("gradient_boosting", "tree"),
    ):
        model, _, _, _, _ = train.train_model(demo_df, model_type)
        assert explainability.explainer_kind(model) == expected


def test_pipeline_is_split_into_estimator_and_transform(demo_df):
    model, _, _, _, _ = train.train_model(demo_df, "logistic_regression")
    estimator, transform = explainability.split_pipeline(model)
    assert estimator.__class__.__name__ == "LogisticRegression"
    assert transform is not None
    assert hasattr(transform, "transform")


def test_global_importance_reports_the_explainer(demo_df):
    model, _, X_test, _, _ = train.train_model(demo_df, "random_forest")
    meta = explainability.global_importance_with_meta(model, X_test)
    assert meta["explainer"] == "tree"
    assert meta["n_background"] > 0
    assert meta["n_explained"] == meta["n_background"]
    assert set(meta["global_importance"]) == set(X_test.columns)


def test_local_explanation_reports_the_explainer(demo_df):
    model, _, X_test, _, _ = train.train_model(demo_df, "logistic_regression")
    local = explainability.local_explanation(model, X_test, row_index=1)
    assert local["explainer"] == "linear"
    assert set(local["contributions"]) == set(X_test.columns)


def test_explainer_is_not_the_slow_permutation_fallback(demo_df):
    """A pipeline must never end up on the permutation explainer."""
    for model_type in train.MODEL_TYPES:
        model, _, _, _, _ = train.train_model(demo_df, model_type)
        assert explainability.explainer_kind(model) != "permutation"


def test_tree_explainer_is_orders_of_magnitude_faster(demo_df):
    """RC-01: the permutation explainer took 234 s on a 99-feature dataset."""
    model, _, X_test, _, _ = train.train_model(demo_df, "random_forest")
    background = X_test.head(80)

    start = time.perf_counter()
    explainability.global_importance(model, background, sample=80)
    tree_seconds = time.perf_counter() - start

    assert tree_seconds < 5.0, f"tree explainer took {tree_seconds:.1f}s"


def test_shap_values_shape_is_two_dimensional(demo_df):
    """(rows, features) — the positive class only."""
    model, _, X_test, _, _ = train.train_model(demo_df, "random_forest")
    adapter = explainability._explainer(model, X_test.head(40))
    values = adapter.shap_values(X_test.head(40))
    assert values.shape == (len(X_test.head(40)), len(X_test.columns))
