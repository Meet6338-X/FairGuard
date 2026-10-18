"""Tests for fairguard.train — models, encoding, save/load."""
import pytest
from joblib import dump

from fairguard import train

ENCODED_COLUMNS = [
    "age", "income_lakhs", "credit_score", "employed_years", "existing_loans",
    "gender_female", "gender_male", "region_rural", "region_urban",
]


def test_make_model():
    assert train.make_model("logistic_regression") is not None
    assert train.make_model("random_forest") is not None
    with pytest.raises(ValueError):
        train.make_model("magic")


def test_encode_features(demo_df):
    X = train.encode_features(demo_df)
    assert list(X.columns) == ENCODED_COLUMNS
    assert len(X) == len(demo_df)


def test_encode_features_aligns_to_columns(demo_df):
    X = train.encode_features(demo_df, columns=["age", "gender_female"])
    assert list(X.columns) == ["age", "gender_female"]


def test_encode_features_needs_feature_columns(demo_df):
    with pytest.raises(ValueError):
        train.encode_features(demo_df.drop(columns=["credit_score"]))


def test_train_model_split(demo_df):
    model, X_train, X_test, y_train, y_test = train.train_model(demo_df, test_size=0.25)
    assert len(X_test) == round(len(demo_df) * 0.25)
    assert len(X_train) + len(X_test) == len(demo_df)
    assert hasattr(model, "predict_proba")


def test_save_load_roundtrip(tmp_path, bundle):
    path = train.save_model(
        bundle["model"], tmp_path / "model.pkl", ENCODED_COLUMNS, "logistic_regression"
    )
    assert path.exists()
    artifact = train.load_model(path)
    assert artifact["model_type"] == "logistic_regression"
    assert artifact["features"] == ENCODED_COLUMNS
    assert hasattr(artifact["model"], "predict_proba")


def test_load_model_rejects_foreign_file(tmp_path):
    path = tmp_path / "not_a_model.pkl"
    dump({"something": "else"}, path)
    with pytest.raises(ValueError):
        train.load_model(path)
