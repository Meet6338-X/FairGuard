"""Tests for fairguard.pipeline — the end-to-end assessment."""
import pytest

from fairguard import pipeline, train


@pytest.fixture(scope="module")
def rf_bundle(demo_df):
    model, X_train, X_test, y_train, y_test = train.train_model(demo_df, "random_forest")
    return {"model": model, "X_test": X_test}


def test_run_assessment_full_record(demo_df, bundle):
    assessment = pipeline.run_assessment(
        df=demo_df.head(150),
        model=bundle["model"],
        features=bundle["X_test"].columns,
        reference=demo_df,
        model_id="demo_model",
        model_version="1.0",
        dataset_id="demo_data",
        run_mlflow=False,
    )
    for key in ("model_id", "model_version", "dataset_id", "sensitive_attribute",
                "performance", "fairness", "privacy", "explainability", "drift",
                "risk", "recommendations", "decision", "reviewer_decision", "assessed_at"):
        assert key in assessment
    assert "mlflow_run_id" not in assessment
    assert assessment["fairness"]["sensitive_attribute"] == "gender"
    assert 0 <= assessment["risk"]["score"] <= 1
    assert assessment["decision"]
    assert assessment["reviewer_decision"] is None


def test_run_assessment_without_reference_or_explainability(demo_df, bundle):
    assessment = pipeline.run_assessment(
        df=demo_df.sample(150, random_state=1),
        model=bundle["model"],
        features=bundle["X_test"].columns,
        run_mlflow=False,
        include_explainability=False,
    )
    assert "drift" not in assessment
    assert "explainability" not in assessment


def test_run_assessment_validation_error(demo_df, bundle):
    with pytest.raises(ValueError):
        pipeline.run_assessment(
            df=demo_df.head(20),  # too small
            model=bundle["model"],
            features=bundle["X_test"].columns,
            run_mlflow=False,
        )


def test_compare_models(demo_df, bundle, rf_bundle):
    results = pipeline.compare_models(
        {
            "logistic_regression": (bundle["model"], bundle["X_test"].columns),
            "random_forest": (rf_bundle["model"], rf_bundle["X_test"].columns),
        },
        df=demo_df.head(150),
        run_mlflow=False,
        include_explainability=False,
    )
    assert set(results) == {"logistic_regression", "random_forest"}
    for name, record in results.items():
        assert record["model_id"] == name
        assert "risk" in record
