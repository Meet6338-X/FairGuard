"""Train, save and load the baseline models that FairGuard assesses."""
from pathlib import Path

import pandas as pd
from joblib import dump, load
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config

LABEL = "approved"

# Features the demo models learn from. `gender` and `age` are kept in on purpose:
# the demo is about *catching* a model that relies on sensitive attributes.
FEATURES = ["age", "income_lakhs", "credit_score", "employed_years", "existing_loans", "gender", "region"]
CATEGORICAL = ["gender", "region"]
MODEL_TYPES = ["logistic_regression", "random_forest", "gradient_boosting"]


def make_model(model_type: str = "logistic_regression"):
    """Create an untrained estimator: median-imputed, and scaled for the linear model."""
    if model_type == "logistic_regression":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE),
        )
    if model_type == "random_forest":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=300, max_depth=12, random_state=config.RANDOM_STATE),
        )
    if model_type == "gradient_boosting":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(random_state=config.RANDOM_STATE),
        )
    raise ValueError(f"unknown model_type: {model_type!r}")


def encode_features(df: pd.DataFrame, columns=None, features=None, categorical=None) -> pd.DataFrame:
    """One-hot encode the given features (defaults to the demo FEATURES), aligned to `columns`."""
    feats = list(features) if features is not None else FEATURES
    cats = list(categorical) if categorical is not None else CATEGORICAL
    cats = [c for c in cats if c in feats]
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"dataset is missing feature columns: {missing}")
    X = pd.get_dummies(df[feats], columns=cats, dtype=float)
    if columns is not None:
        X = X.reindex(columns=list(columns), fill_value=0.0)
    return X


# id-ish and personal-contact columns never make good model features.
AUTO_EXCLUDE = {"applicant_id", "id", "email", "phone", "full_name", "fnlwgt"}


def pick_features(df: pd.DataFrame, label: str, sensitive: str = None, exclude=None) -> list:
    """Cheap automatic feature pick for a new dataset: numeric columns minus ids/PII/label/sensitive."""
    skip = {label, *AUTO_EXCLUDE, *(exclude or [])}
    if sensitive:
        skip.add(sensitive)
    return [c for c in df.select_dtypes(include="number").columns if c not in skip]


def train_model(df, model_type="logistic_regression", test_size=None, label=LABEL, features=None):
    """Fit a model and return (model, X_train, X_test, y_train, y_test).

    `features=None` uses the demo FEATURES; pass `pick_features(df, label, sensitive)`
    for any other dataset (see scripts/train_demo_model.py).
    """
    test_size = test_size or config.TEST_SPLIT
    X = encode_features(df, features=list(features) if features is not None else None)
    y = df[label]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=config.RANDOM_STATE, stratify=y
    )
    model = make_model(model_type).fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def save_model(model, path, feature_columns, model_type: str = "logistic_regression",
               raw_features=None) -> Path:
    """Save model + the feature column list it was trained on, in one artifact.

    `feature_columns` holds the encoded columns handed to the estimator;
    `raw_features` is the pre-encoding column list (defaults to the same list).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dump({"model": model,
          "features": list(feature_columns),
          "raw_features": list(raw_features) if raw_features is not None else list(feature_columns),
          "model_type": model_type}, path)
    return path


def load_model(path) -> dict:
    """Load the artifact written by save_model: {'model', 'features', 'model_type'}."""
    artifact = load(Path(path))
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError(f"{path} is not a FairGuard model artifact")
    return artifact
