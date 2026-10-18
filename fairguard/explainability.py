"""Model explanations with SHAP — global importance and single-prediction views."""
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def _shap_values(explainer, X: pd.DataFrame) -> np.ndarray:
    """SHAP values of the positive class, across shap versions/return types."""
    if hasattr(explainer, "shap_values"):
        values = explainer.shap_values(X)
    else:  # e.g. ExactExplainer returned by shap.Explainer(model.predict_proba, ...)
        values = explainer(X).values
    if isinstance(values, list):           # older API: [class_0, class_1]
        values = values[1]
    if hasattr(values, "values"):          # shap 0.45+: Explanation object
        values = values.values
    values = np.asarray(values)
    if values.ndim == 3:                   # (rows, features, classes)
        values = values[:, :, 1]
    return values


def _explainer(model, background: pd.DataFrame):
    """Pick the cheapest explainer that fits the model."""
    if isinstance(model, LogisticRegression):
        return shap.LinearExplainer(model, background)
    if isinstance(model, RandomForestClassifier):
        return shap.TreeExplainer(model)
    return shap.Explainer(model.predict_proba, background)


def global_importance(model, X: pd.DataFrame, sample: int = 300) -> dict:
    """Mean |SHAP| per feature, largest first."""
    background = X.sample(min(sample, len(X)), random_state=0)
    values = _shap_values(_explainer(model, background), background)
    importance = np.abs(values).mean(axis=0)
    pairs = sorted(zip(X.columns, importance), key=lambda kv: kv[1], reverse=True)
    return {str(col): round(float(val), 4) for col, val in pairs}


def local_explanation(model, X: pd.DataFrame, row_index: int = 0) -> dict:
    """Feature contributions for one prediction (positive pushes toward 'approved')."""
    background = X.sample(min(300, len(X)), random_state=0)
    values = _shap_values(_explainer(model, background), X.iloc[[row_index]])[0]
    probability = float(model.predict_proba(X.iloc[[row_index]])[0, 1])
    pairs = sorted(zip(X.columns, values), key=lambda kv: abs(kv[1]), reverse=True)
    return {
        "row_index": int(row_index),
        "probability": round(probability, 4),
        "contributions": {str(col): round(float(val), 4) for col, val in pairs},
    }
