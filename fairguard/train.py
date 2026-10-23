"""Train, save and load the baseline models that FairGuard assesses.

Root causes fixed here
-----------------------
* **RC-03** `pick_features` used to keep numeric columns only, so every
  categorical column of a real dataset (adult: 9 of them) was silently
  dropped — a large loss of signal *and* a fairness blind spot, because
  categorical columns are often the proxies for the sensitive attribute.
* **RC-10** `train_test_split(..., stratify=y)` blew up with a confusing
  ``ValueError`` when a class had fewer than two members.
* **RC-19** No class-imbalance handling, which hurts recall on the minority
  class and therefore hurts equalized odds.
* **RC-02** The model artifact now carries the hold-out split, so an
  assessment can be scored out-of-sample instead of on training rows.
"""
from pathlib import Path

import numpy as np
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

# Categorical columns are one-hot encoded when they have at most this many
# distinct values; anything richer is dropped (it would explode the feature
# space and is usually a free-text field).
MAX_CATEGORICAL_CARDINALITY = 20


def make_model(model_type: str = "logistic_regression", balance_classes: bool = True):
    """Create an untrained estimator: median-imputed, and scaled for the linear model.

    ``balance_classes=True`` uses balanced class weights.  That matters for the
    fairness numbers too: a model that ignores the minority class usually has
    a much worse false-negative rate for that group, which is exactly what
    equalized odds measures.
    """
    weight = "balanced" if balance_classes else None
    if model_type == "logistic_regression":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight=weight,
                               random_state=config.RANDOM_STATE),
        )
    if model_type == "random_forest":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=300, max_depth=12, class_weight=weight,
                                    random_state=config.RANDOM_STATE),
        )
    if model_type == "gradient_boosting":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(class_weight=weight,
                                           random_state=config.RANDOM_STATE),
        )
    raise ValueError(f"unknown model_type: {model_type!r}")


def encode_features(df: pd.DataFrame, columns=None, features=None, categorical=None) -> pd.DataFrame:
    """One-hot encode the given features (defaults to the demo FEATURES), aligned to `columns`.

    `categorical` defaults to every *low-cardinality* non-numeric column of the
    selected features (RC-03).  Pass an explicit list to override.
    """
    feats = list(features) if features is not None else FEATURES
    if categorical is not None:
        cats = [c for c in list(categorical) if c in feats]
    else:
        cats = [c for c in feats
                if c in df.columns
                and not pd.api.types.is_numeric_dtype(df[c])
                and df[c].nunique(dropna=True) <= MAX_CATEGORICAL_CARDINALITY]
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"dataset is missing feature columns: {missing}")
    X = pd.get_dummies(df[feats], columns=cats, dtype=float)
    if columns is not None:
        X = X.reindex(columns=list(columns), fill_value=0.0)
    return X


# id-ish and personal-contact columns never make good model features.
AUTO_EXCLUDE = {"applicant_id", "id", "email", "phone", "full_name", "fnlwgt"}


def label_duplicate_columns(df: pd.DataFrame, label: str,
                             max_categories: int = 50) -> list:
    """Columns that are a 1:1 encoding of the label — i.e. leakage (RC-23).

    A dataset often ships the raw label next to a derived binary one
    (`adult.csv` has `income` and `income_gt_50k`).  As soon as categorical
    columns became usable features, such a column lets the model read the
    answer straight off the feature: the Adult model reached a hold-out
    accuracy of **1.000**, which is not a result, it is a leak.

    A column is reported when its distinct values map bijectively onto the
    label's distinct values, so each column value determines the label and
    vice versa.  Columns with more distinct values than `max_categories` are
    ignored — a numeric id is not a leak, it is noise.
    """
    if label not in df.columns:
        return []
    y = df[label]
    n_label = y.nunique(dropna=True)
    if n_label < 2 or n_label > max_categories:
        return []
    duplicates = []
    for column in df.columns:
        if column == label:
            continue
        series = df[column]
        if series.nunique(dropna=True) > max_categories:
            continue
        pairs = pd.DataFrame({"column": series, "label": y}).dropna().drop_duplicates()
        if len(pairs) == n_label and pairs["column"].nunique() == n_label:
            duplicates.append(column)
    return duplicates


def pick_features(df: pd.DataFrame, label: str, sensitive: str = None, exclude=None,
                  include_categorical: bool = True) -> list:
    """Automatic feature pick for a new dataset.

    Keeps numeric columns *and* (RC-03) low-cardinality categorical columns,
    minus ids / PII / label / the sensitive attribute / label duplicates
    (RC-23).
    """
    skip = {label, *AUTO_EXCLUDE, *(exclude or [])}
    if sensitive:
        skip.add(sensitive)
    # RC-23: a column that is a copy of the label is not a feature.
    skip.update(label_duplicate_columns(df, label))
    numeric = [c for c in df.select_dtypes(include="number").columns if c not in skip]
    if not include_categorical:
        return numeric
    categorical = [
        c for c in df.columns
        if c not in skip
        and not pd.api.types.is_numeric_dtype(df[c])
        and df[c].nunique(dropna=True) <= MAX_CATEGORICAL_CARDINALITY
    ]
    return numeric + categorical


def split_data(X: pd.DataFrame, y: pd.Series, test_size=None, random_state=None):
    """Stratified train/test split that degrades gracefully (RC-10).

    Returns ``(X_train, X_test, y_train, y_test, stratified)``.  When a class
    has fewer than two members, stratification is impossible; instead of
    crashing with an opaque sklearn error the split falls back to unstratified
    and reports ``stratified=False``.
    """
    test_size = test_size or config.TEST_SPLIT
    random_state = config.RANDOM_STATE if random_state is None else random_state
    counts = pd.Series(y).value_counts()
    can_stratify = len(counts) >= 2 and int(counts.min()) >= 2
    stratify = y if can_stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return X_train, X_test, y_train, y_test, can_stratify


def train_model(df, model_type="logistic_regression", test_size=None, label=LABEL,
                features=None, balance_classes=True):
    """Fit a model and return (model, X_train, X_test, y_train, y_test).

    `features=None` uses the demo FEATURES; pass `pick_features(df, label, sensitive)`
    for any other dataset (see scripts/train_demo_model.py).

    RC-23: a column that is a copy of the label is never trained on, even when
    the caller hands the feature list in explicitly.
    """
    test_size = test_size or config.TEST_SPLIT
    if features is not None:
        # The label is never a feature, and neither is a copy of it (RC-23).
        leaked = {label, *label_duplicate_columns(df, label)}
        features = [f for f in features if f not in leaked]
    X = encode_features(df, features=list(features) if features is not None else None)
    y = df[label]
    X_train, X_test, y_train, y_test, _ = split_data(X, y, test_size=test_size)
    model = make_model(model_type, balance_classes=balance_classes).fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def save_model(model, path, feature_columns, model_type: str = "logistic_regression",
               raw_features=None, holdout=None, meta=None) -> Path:
    """Save model + the feature column list it was trained on, in one artifact.

    `feature_columns` holds the encoded columns handed to the estimator;
    `raw_features` is the pre-encoding column list (defaults to the same list).
    `holdout` (RC-02) is an optional dict with the positional indices of the
    hold-out rows inside the training CSV, so an assessment can be scored
    out-of-sample.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dump({"model": model,
          "features": list(feature_columns),
          "raw_features": list(raw_features) if raw_features is not None else list(feature_columns),
          "model_type": model_type,
          "holdout": holdout,
          "meta": dict(meta or {})}, path)
    return path


def load_model(path) -> dict:
    """Load the artifact written by save_model: {'model', 'features', 'model_type'}."""
    artifact = load(Path(path))
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError(f"{path} is not a FairGuard model artifact")
    artifact.setdefault("holdout", None)
    artifact.setdefault("meta", {})
    return artifact


def make_holdout(df: pd.DataFrame, test_size=None, label=LABEL, features=None,
                 random_state=None) -> dict:
    """Compute (and return) the hold-out split of ``df`` without training.

    Mirrors `train_model`'s split so the two stay in sync: the same
    ``random_state``/``test_size``/``stratify`` rules produce the same rows.
    The stored indices are **positional** (row 0 = first data row of the CSV),
    so they stay valid when the frame is re-read.
    """
    test_size = test_size or config.TEST_SPLIT
    frame = df.reset_index(drop=True)
    if features is not None:
        leaked = {label, *label_duplicate_columns(frame, label)}
        features = [f for f in features if f not in leaked]
    X = encode_features(frame, features=list(features) if features is not None else None)
    y = frame[label]
    _, X_test, _, y_test, stratified = split_data(X, y, test_size=test_size,
                                                  random_state=random_state)
    return {
        "test_size": float(test_size),
        "random_state": config.RANDOM_STATE if random_state is None else random_state,
        "stratified": bool(stratified),
        "test_index": [int(i) for i in X_test.index],
        "n_test": int(len(X_test)),
        "label_positives": int(pd.Series(y_test).sum()),
    }


def holdout_rows(df: pd.DataFrame, holdout: dict) -> pd.DataFrame:
    """Slice the hold-out rows out of ``df`` using a `make_holdout` record."""
    if not holdout:
        return df
    index = holdout.get("test_index") or []
    if not index:
        return df
    frame = df.reset_index(drop=True)
    keep = [i for i in index if 0 <= i < len(frame)]
    if not keep:
        return df
    return frame.iloc[keep].reset_index(drop=True)
