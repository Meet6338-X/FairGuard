"""Tests for the hold-out evaluation scope — RC-02.

The assessment used to score the model on the *whole* dataset, including the
rows it was trained on.  A trained model now carries its hold-out split and
the assessment states which scope produced the numbers.
"""
import pandas as pd
import pytest

from fairguard import config, pipeline, train


def test_make_holdout_matches_the_training_split(demo_df):
    """The stored hold-out must be exactly the rows `train_model` held back."""
    model, X_train, X_test, y_train, y_test = train.train_model(demo_df, "logistic_regression")
    holdout = train.make_holdout(demo_df, label=train.LABEL)

    assert holdout["n_test"] == len(X_test)
    assert holdout["stratified"] is True
    assert len(holdout["test_index"]) == len(X_test)

    rows = train.holdout_rows(demo_df, holdout)
    assert len(rows) == len(X_test)
    # The same rows, in the same order.
    assert list(rows.index) == list(range(len(rows)))


def test_holdout_rows_are_exactly_the_stored_positions(demo_df):
    """The stored indices must point at the rows the hold-out contains.

    ``train_test_split`` returns the held-out rows in the order of its random
    permutation, and `holdout_rows` preserves that order so the assessment
    lines up row-for-row with the training split.
    """
    holdout = train.make_holdout(demo_df, label=train.LABEL)
    rows = train.holdout_rows(demo_df, holdout)
    index = holdout["test_index"]
    assert len(rows) == len(index)
    assert rows[train.LABEL].tolist() == demo_df.iloc[index][train.LABEL].tolist()


def test_holdout_matches_the_rows_train_model_held_back(demo_df):
    """The critical invariant: the assessment scores exactly the test split."""
    model, X_train, X_test, y_train, y_test = train.train_model(
        demo_df, "logistic_regression")
    holdout = train.make_holdout(demo_df, label=train.LABEL)
    rows = train.holdout_rows(demo_df, holdout)
    assert len(rows) == len(X_test)
    assert rows[train.LABEL].tolist() == list(y_test)


def test_holdout_rows_are_disjoint_from_the_training_rows(demo_df):
    """No row may be both trained on and scored."""
    model, X_train, X_test, y_train, y_test = train.train_model(
        demo_df, "logistic_regression")
    holdout = train.make_holdout(demo_df, label=train.LABEL)
    test_positions = set(holdout["test_index"])
    train_positions = set(range(len(demo_df))) - test_positions
    assert test_positions & train_positions == set()
    assert len(train_positions) == len(X_train)
    assert len(test_positions) == len(X_test)


def test_artifact_round_trips_the_holdout(tmp_path, demo_df):
    model, X_train, _, _, _ = train.train_model(demo_df, "logistic_regression")
    holdout = train.make_holdout(demo_df, label=train.LABEL)
    path = tmp_path / "model.pkl"
    train.save_model(model, path, X_train.columns, "logistic_regression",
                     raw_features=train.FEATURES, holdout=holdout)
    artifact = train.load_model(path)
    assert artifact["holdout"]["n_test"] == holdout["n_test"]
    assert artifact["holdout"]["test_index"] == holdout["test_index"]


def test_assessment_on_the_holdout_reports_the_scope(demo_df):
    model, X_train, _, _, _ = train.train_model(demo_df, "logistic_regression")
    holdout = train.make_holdout(demo_df, label=train.LABEL)
    eval_rows = train.holdout_rows(demo_df, holdout)

    assessment = pipeline.run_assessment(
        df=demo_df,
        model=model,
        features=X_train.columns,
        label=train.LABEL,
        sensitive="gender",
        eval_df=eval_rows,
        run_mlflow=False,
        include_explainability=False,
    )
    scope = assessment["evaluation"]
    assert scope["scope"] == "holdout"
    assert scope["in_sample"] is False
    assert scope["n_rows"] == len(eval_rows)
    assert scope["n_rows"] < scope["n_dataset_rows"]


def test_assessment_without_a_holdout_is_marked_in_sample(demo_df):
    model, X_train, _, _, _ = train.train_model(demo_df, "logistic_regression")
    assessment = pipeline.run_assessment(
        df=demo_df,
        model=model,
        features=X_train.columns,
        label=train.LABEL,
        sensitive="gender",
        run_mlflow=False,
        include_explainability=False,
    )
    scope = assessment["evaluation"]
    assert scope["scope"] == "full_dataset"
    assert scope["in_sample"] is True
    assert scope["n_rows"] == scope["n_dataset_rows"]


def test_holdout_metrics_differ_from_in_sample_metrics(demo_df):
    """Out-of-sample numbers must not be the in-sample numbers."""
    model, X_train, _, _, _ = train.train_model(demo_df, "logistic_regression")
    holdout = train.make_holdout(demo_df, label=train.LABEL)
    eval_rows = train.holdout_rows(demo_df, holdout)

    common = dict(model=model, features=X_train.columns, label=train.LABEL,
                  sensitive="gender", run_mlflow=False, include_explainability=False)
    in_sample = pipeline.run_assessment(df=demo_df, **common)
    out_of_sample = pipeline.run_assessment(df=demo_df, eval_df=eval_rows, **common)

    assert in_sample["evaluation"]["n_rows"] != out_of_sample["evaluation"]["n_rows"]
    assert (in_sample["performance"]["accuracy"]
            != out_of_sample["performance"]["accuracy"])
