"""RC-23 hardening — leaks must also surface at *assessment* time.

The audit caught the label duplicate in adult.csv while a model was
*training*.  But artifacts trained before that fix may still be lying
around, so the assessment itself has to flag a model that carries a
1:1 copy of the label instead of reporting a beautiful ~1.0 accuracy.
"""
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fairguard import pipeline, train


def _toy_frame(n: int = 60) -> pd.DataFrame:
    """Frame with a label and a column that is exactly the label."""
    return pd.DataFrame({
        "age": range(n),
        "score": [i / n for i in range(n)],
        "answer": [i % 2 for i in range(n)],          # the label itself
        "answer_copy": ["yes" if i % 2 else "no" for i in range(n)],  # 1:1 duplicate
        "noise": [(i * 37) % 11 for i in range(n)],
        "gender": ["male" if i % 2 else "female" for i in range(n)],
        "approved": [i % 2 for i in range(n)],
    })


def _model(df: pd.DataFrame, label: str = "approved", leak: str | None = None):
    """Train a model the way a *stale* artifact would have been trained.

    ``leak=None`` reproduces a clean, guarded artifact; any other value
    is added to the raw features first (the RC-23 pre-guard behaviour).
    """
    raw = list(train.pick_features(df, label, "gender"))
    if leak:
        raw = raw + [leak]          # the exact bug the guard now prevents
    X = train.encode_features(df, features=raw)
    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=2000),
    )
    model.fit(X, df[label])
    return model, raw, X.columns.tolist()


def test_assessment_flags_leak_from_raw_features():
    df = _toy_frame()
    model, raw, encoded = _model(df, leak="answer_copy")
    assessment = pipeline.run_assessment(
        df=df,
        model=model,
        features=encoded,
        raw_features=raw,
        label="approved",
        sensitive="gender",
        include_explainability=False,
        run_mlflow=False,
    )
    assert set(assessment["leakage"]["columns"]) == {"answer_copy"}
    assert any("Leakage" in rec for rec in assessment["recommendations"])


def test_assessment_flags_label_used_as_feature():
    df = _toy_frame()
    model, raw, encoded = _model(df, leak="answer")
    assessment = pipeline.run_assessment(
        df=df,
        model=model,
        features=encoded,
        raw_features=raw,
        label="approved",
        sensitive="gender",
        include_explainability=False,
        run_mlflow=False,
    )
    assert "answer" in assessment["leakage"]["columns"]
    assert any("Leakage" in rec for rec in assessment["recommendations"])
    # A leak makes any scoring scope in-sample knowledge.
    assert assessment["evaluation"]["in_sample"] is True


def test_assessment_flags_when_the_model_score_is_perfect():
    """The realistic stale-artifact scenario: leak -> accuracy ~1.0."""
    df = _toy_frame()
    model, raw, encoded = _model(df, leak="answer_copy")
    assessment = pipeline.run_assessment(
        df=df,
        model=model,
        features=encoded,
        raw_features=raw,
        label="approved",
        sensitive="gender",
        include_explainability=False,
        run_mlflow=False,
    )
    assert assessment["performance"]["accuracy"] > 0.9
    assert "leakage" in assessment


def test_clean_assessment_has_no_leak_block():
    df = _toy_frame()
    model, raw, encoded = _model(df)
    assessment = pipeline.run_assessment(
        df=df,
        model=model,
        features=encoded,
        raw_features=raw,
        label="approved",
        sensitive="gender",
        include_explainability=False,
        run_mlflow=False,
    )
    assert "leakage" not in assessment
    assert all(not rec.startswith("Leakage") for rec in assessment["recommendations"])
