"""Data / model drift detection — Population Stability Index and KS test (SRS).

Root cause fixed here
----------------------
**RC-06** ``drift_risk`` used the *mean* per-feature severity.  One feature in
MAJOR drift among twenty therefore scored 0.05 — a critical shift in an
important feature read as "no risk".  The aggregation is now importance
aware: a drift in a feature the model relies on counts for much more than a
drift in a feature it ignores, and the worst single feature always keeps
some weight.
"""
import numpy as np
import pandas as pd
from scipy import stats

from . import config

EPS = 1e-6
# Identifier / PII columns are not model inputs; drift on them is meaningless.
EXCLUDED = {"applicant_id", "email", "phone", "full_name", "id"}


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index between two numeric samples."""
    ref = pd.Series(reference).dropna().astype(float)
    cur = pd.Series(current).dropna().astype(float)
    if ref.empty or cur.empty:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # constant-ish feature: nothing meaningful to compare
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.clip(np.histogram(ref, bins=edges)[0] / len(ref), EPS, None)
    cur_pct = np.clip(np.histogram(cur, bins=edges)[0] / len(cur), EPS, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    """PSI for categorical features, comparing value proportions."""
    ref = reference.value_counts(normalize=True)
    cur = current.value_counts(normalize=True)
    index = ref.index.union(cur.index)
    ref = ref.reindex(index, fill_value=0).clip(lower=EPS)
    cur = cur.reindex(index, fill_value=0).clip(lower=EPS)
    return float(((cur - ref) * np.log(cur / ref)).sum())


def ks(reference, current) -> tuple:
    """KS statistic and p-value between two numeric samples."""
    stat, pvalue = stats.ks_2samp(reference.dropna(), current.dropna())
    return float(stat), float(pvalue)


def psi_level(value: float) -> str:
    """Bucket a PSI value using the SRS thresholds."""
    if value < config.PSI_MODERATE:
        return "stable"
    if value < config.PSI_MAJOR:
        return "moderate"
    return "major"


def assess_drift(reference: pd.DataFrame, current: pd.DataFrame, features=None) -> dict:
    """Per-feature PSI + KS between reference and current data."""
    default = [c for c in reference.columns if c in current.columns and c not in EXCLUDED]
    columns = features or default
    result = {}
    for column in columns:
        ref_col, cur_col = reference[column], current[column]
        numeric = pd.api.types.is_numeric_dtype(ref_col) and ref_col.nunique() > 10
        if numeric:
            value = psi(ref_col, cur_col)
            stat, pvalue = ks(ref_col, cur_col)
        else:
            value = categorical_psi(ref_col, cur_col)
            stat, pvalue = None, None
        result[column] = {
            "psi": round(value, 4),
            "psi_level": psi_level(value),
            "ks_stat": round(stat, 4) if stat is not None else None,
            "ks_pvalue": round(pvalue, 6) if pvalue is not None else None,
            "missing_pct_shift": round(float(cur_col.isna().mean() - ref_col.isna().mean()) * 100, 2),
        }
    return result


def aggregate_importance(importance: dict, raw_features=None) -> dict:
    """Fold encoded (one-hot) feature importance back onto raw column names.

    SHAP importance is reported per *encoded* column (``gender_female``),
    while drift is measured per *raw* column (``gender``).  This maps the
    first onto the second so the two can be combined.

    ``raw_features`` (the pre-encoding column list) is used when available.
    Without it a column is only treated as a one-hot expansion when at least
    one other column shares its prefix — otherwise ``credit_score`` would be
    mistaken for an expansion of ``credit``.
    """
    if not importance:
        return {}
    totals: dict[str, float] = {}
    for column, value in importance.items():
        try:
            weight = float(value)
        except (TypeError, ValueError):
            continue
        name = str(column)
        base = name
        if raw_features:
            if name in raw_features:
                base = name
            else:
                match = next((r for r in raw_features if name.startswith(f"{r}_")), None)
                base = match or name
        elif "_" in name:
            prefix = name.rsplit("_", 1)[0]
            siblings = [c for c in importance if str(c) != name and str(c).startswith(f"{prefix}_")]
            if siblings:
                base = prefix
        totals[base] = totals.get(base, 0.0) + weight
    return totals


def drift_risk(result: dict, importance: dict = None, raw_features=None) -> float:
    """0..1 drift risk.

    ``importance`` (mean |SHAP| per feature) is optional.  When it is given,
    the aggregate is weighted by how much the model relies on each feature;
    the worst single feature always keeps part of the score so a critical
    shift is never averaged away.
    """
    if not result:
        return 0.0
    severity = {"stable": 0.0, "moderate": 0.5, "major": 1.0}
    scores = {column: float(severity.get(entry.get("psi_level"), 0.0))
              for column, entry in result.items()}
    peak = max(scores.values()) if scores else 0.0
    flat = float(np.mean(list(scores.values()))) if scores else 0.0

    weight = config.DRIFT_IMPORTANCE_WEIGHT
    aggregate = flat
    if importance:
        totals = aggregate_importance(importance, raw_features=raw_features)
        shared = {column: totals[column] for column in scores if column in totals}
        if shared:
            total_weight = sum(shared.values())
            if total_weight > 0:
                aggregate = sum(scores[column] * w for column, w in shared.items()) / total_weight
        else:
            # No overlap between the two naming schemes — fall back to the flat
            # mean rather than pretending the weighting worked.
            weight = 0.5
    else:
        # Without importance the flat mean alone would dilute a critical
        # shift, so half the score is reserved for the worst feature.
        weight = 0.5

    risk = weight * aggregate + (1.0 - weight) * peak
    return float(min(max(risk, 0.0), 1.0))
