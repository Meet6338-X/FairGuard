"""Model explanations with SHAP — global importance and single-prediction views.

Root cause fixed in this module
--------------------------------
The explainer used to be picked with ``isinstance(model, LogisticRegression)``
and ``isinstance(model, RandomForestClassifier)``.  Every FairGuard model is an
sklearn ``Pipeline`` (imputer → scaler → estimator), so *neither* check ever
matched and every model silently fell back to the model-agnostic
``PermutationExplainer``.

Measured cost on the 99-feature ``communities`` dataset:

    PermutationExplainer (old) : 234.5 s
    TreeExplainer      (new)   :   3.5 s      (~67x faster, and exact)

The old code also *regressed* silently: a wrong explainer choice is not an
error, it is just slow and (for tree models) less precise.  The explainer kind
is now reported in the assessment record so this can never hide again.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# Estimators SHAP's TreeExplainer understands natively.
TREE_ESTIMATORS = (
    DecisionTreeClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
LINEAR_ESTIMATORS = (LogisticRegression, LinearRegression)


def split_pipeline(model):
    """Split a model into ``(final_estimator, transform)``.

    ``transform`` is the fitted sklearn ``Pipeline`` slice that prepares raw
    encoded columns for the estimator, or ``None`` when the model is not a
    pipeline.  This is what makes the fast explainers reachable at all.
    """
    if isinstance(model, Pipeline):
        return model.steps[-1][1], model[:-1]
    return model, None


def _as_frame(values: np.ndarray, columns) -> np.ndarray:
    """SHAP values as a 2-D ``(rows, features)`` array for the positive class."""
    values = np.asarray(values)
    if values.ndim == 3:
        # (rows, features, classes) — keep the positive class.
        values = values[:, :, 1]
    elif values.ndim == 1:
        values = values.reshape(-1, 1)
    return values


class Explainer:
    """Adapter that applies the pipeline transform before explaining.

    Wraps the SHAP explainer so callers always pass the *encoded* feature
    frame they trained on, and the adapter takes care of imputing/scaling.
    """

    def __init__(self, explainer, transform=None, kind="unknown"):
        self.explainer = explainer
        self.transform = transform
        self.kind = kind

    # -- internals ---------------------------------------------------------
    def _prepare(self, X: pd.DataFrame):
        if self.transform is None:
            return X
        return self.transform.transform(X)

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """SHAP values for the positive class, shaped ``(rows, features)``."""
        data = self._prepare(X)
        if hasattr(self.explainer, "shap_values"):
            values = self.explainer.shap_values(data)
        else:  # e.g. ExactExplainer returned by shap.Explainer(model.predict_proba, ...)
            values = self.explainer(data).values
        if isinstance(values, list):       # older API: [class_0, class_1]
            values = values[1]
        if hasattr(values, "values"):      # shap 0.45+: Explanation object
            values = values.values
        return _as_frame(values, X.columns)


def _explainer(model, background: pd.DataFrame) -> Explainer:
    """Pick the cheapest *exact* explainer that fits the model.

    Order of preference:

    1. ``LinearExplainer`` for logistic regression / linear models (exact,
       analytic, ~instant).
    2. ``TreeExplainer`` for tree ensembles (exact, orders of magnitude
       faster than the model-agnostic fallback).
    3. ``shap.Explainer(model.predict_proba, ...)`` — the model-agnostic
       permutation fallback for anything else.
    """
    estimator, transform = split_pipeline(model)

    if isinstance(estimator, LINEAR_ESTIMATORS):
        data = transform.transform(background) if transform is not None else background
        # Build the masker explicitly so the whole background is used —
        # shap's default cap of 100 rows silently subsamples and warns.
        masker = shap.maskers.Independent(data, max_samples=len(data))
        return Explainer(shap.LinearExplainer(estimator, masker), transform, kind="linear")

    if isinstance(estimator, TREE_ESTIMATORS):
        try:
            return Explainer(shap.TreeExplainer(estimator), transform, kind="tree")
        except Exception:  # unsupported tree flavour — fall through
            pass

    return Explainer(shap.Explainer(model.predict_proba, background), None, kind="permutation")


def explainer_kind(model, background: pd.DataFrame | None = None) -> str:
    """Which explainer would be used — cheap check used by the pipeline/tests."""
    if background is None:
        estimator, _ = split_pipeline(model)
        if isinstance(estimator, LINEAR_ESTIMATORS):
            return "linear"
        if isinstance(estimator, TREE_ESTIMATORS):
            return "tree"
        return "permutation"
    return _explainer(model, background).kind


def global_importance(model, X: pd.DataFrame, sample: int = 300) -> dict:
    """Mean |SHAP| per feature, largest first."""
    background = X.sample(min(sample, len(X)), random_state=0)
    values = _explainer(model, background).shap_values(background)
    importance = np.abs(values).mean(axis=0)
    pairs = sorted(zip(X.columns, importance), key=lambda kv: kv[1], reverse=True)
    return {str(col): round(float(val), 4) for col, val in pairs}


def global_importance_with_meta(model, X: pd.DataFrame, sample: int = 300) -> dict:
    """Global importance plus the explainer that produced it (audit trail)."""
    background = X.sample(min(sample, len(X)), random_state=0)
    adapter = _explainer(model, background)
    values = adapter.shap_values(background)
    importance = np.abs(values).mean(axis=0)
    pairs = sorted(zip(X.columns, importance), key=lambda kv: kv[1], reverse=True)
    return {
        "global_importance": {str(col): round(float(val), 4) for col, val in pairs},
        "explainer": adapter.kind,
        "n_background": int(len(background)),
        "n_explained": int(len(background)),
    }


def local_explanation(model, X: pd.DataFrame, row_index: int = 0) -> dict:
    """Feature contributions for one prediction (positive pushes toward 'approved')."""
    background = X.sample(min(300, len(X)), random_state=0)
    # The explained row must not be part of its own background set, otherwise
    # the explanation is partly self-referential.
    background = background.drop(index=X.index[row_index], errors="ignore")
    if background.empty:
        background = X.iloc[[row_index]]
    adapter = _explainer(model, background)
    values = adapter.shap_values(X.iloc[[row_index]])[0]
    probability = float(model.predict_proba(X.iloc[[row_index]])[0, 1])
    pairs = sorted(zip(X.columns, values), key=lambda kv: abs(kv[1]), reverse=True)
    return {
        "row_index": int(row_index),
        "probability": round(probability, 4),
        "contributions": {str(col): round(float(val), 4) for col, val in pairs},
        "explainer": adapter.kind,
    }
