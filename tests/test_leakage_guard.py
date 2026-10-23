"""RC-23 — a column that is a copy of the label must never be a feature.

`data/adult.csv` ships both `income` (">50K" / "<=50K") and the derived
`income_gt_50k` label.  As soon as categorical columns became usable
features, the model could read the answer straight off the feature and the
Adult hold-out accuracy jumped to 1.000 — not a result, a leak.
"""
import pandas as pd

from fairguard import train


def _adult_like():
    return pd.DataFrame({
        "age": [25, 45, 33, 51, 29, 60],
        "income": ["<=50K", ">50K", "<=50K", ">50K", "<=50K", ">50K"],
        "income_gt_50k": [0, 1, 0, 1, 0, 1],
        "sex": ["M", "F", "M", "M", "F", "M"],
    })


def test_label_duplicate_columns_finds_the_copy():
    df = _adult_like()
    assert "income" in train.label_duplicate_columns(df, "income_gt_50k")


def test_label_duplicate_columns_ignores_unrelated_columns():
    df = _adult_like()
    duplicates = train.label_duplicate_columns(df, "income_gt_50k")
    assert "age" not in duplicates
    assert "sex" not in duplicates
    assert "income_gt_50k" not in duplicates


def test_pick_features_drops_the_label_copy():
    df = _adult_like()
    features = train.pick_features(df, "income_gt_50k", sensitive="sex")
    assert "income" not in features
    assert "income_gt_50k" not in features
    assert "sex" not in features
    assert "age" in features


def test_train_model_refuses_an_explicitly_passed_leak():
    """Even when the caller hands the leaky column in, it must not be trained on."""
    df = _adult_like()
    model, X_train, _, _, _ = train.train_model(
        df, "logistic_regression", label="income_gt_50k",
        features=["age", "income", "income_gt_50k"],
    )
    assert not any(c.startswith("income") for c in X_train.columns)
    # without the leak the model cannot be perfect on unseen rows
    assert len(X_train) < len(df)


def test_a_high_cardinality_id_is_not_treated_as_a_leak():
    df = pd.DataFrame({
        "row_id": list(range(50)),
        "label": [0, 1] * 25,
    })
    assert train.label_duplicate_columns(df, "label") == []
