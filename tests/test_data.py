"""Tests for fairguard.data — loading, validation, profiling."""
from fairguard import data


def test_load_csv_roundtrip(tmp_path, demo_df):
    path = tmp_path / "demo.csv"
    demo_df.to_csv(path, index=False)
    loaded = data.load_csv(path)
    assert list(loaded.columns) == list(demo_df.columns)
    assert len(loaded) == len(demo_df)


def test_add_age_group(demo_df):
    out = data.add_age_group(demo_df)
    assert "age_group" in out.columns
    assert set(out["age_group"].unique()) <= {"<30", "30-45", ">45"}
    assert "age_group" not in demo_df.columns  # original untouched


def test_validate_dataset_ok(demo_df):
    assert data.validate_dataset(demo_df, ["gender"]) == []


def test_validate_dataset_reports_problems(demo_df):
    problems = data.validate_dataset(demo_df.drop(columns=["approved"]), ["gender"])
    assert any("approved" in p for p in problems)

    problems = data.validate_dataset(demo_df.head(10), ["gender"])
    assert any("small" in p for p in problems)


def test_validate_dataset_accepts_encoded_features(demo_df):
    # 'gender_female' is fine because 'gender' exists in the dataset
    assert data.validate_dataset(
        demo_df, ["gender"], model_features=["age", "gender_female", "region_urban"]
    ) == []
    problems = data.validate_dataset(demo_df, ["gender"], model_features=["nonsense_feature"])
    assert any("nonsense_feature" in p for p in problems)


def test_profile_dataset(demo_df):
    profile = data.profile_dataset(demo_df)
    assert profile["rows"] == len(demo_df)
    assert profile["columns"] == demo_df.shape[1]
    assert "credit_score" in profile["numeric_summary"]
    assert "gender" in profile["categorical_summary"]
    assert profile["numeric_summary"]["credit_score"]["min"] <= profile["numeric_summary"]["credit_score"]["mean"]
