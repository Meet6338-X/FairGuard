"""Data / model drift detection — Population Stability Index and KS test (SRS)."""
import numpy as np
import pandas as pd
from scipy import stats

from . import config

EPS = 1e-6
# Identifier / PII columns are not model inputs; drift on them is meaningless.
EXCLUDED = {"applicant_id", "email", "phone", "full_name"}


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


def drift_risk(result: dict) -> float:
    """Mean per-feature severity: stable 0, moderate 0.5, major 1.0."""
    if not result:
        return 0.0
    severity = {"stable": 0.0, "moderate": 0.5, "major": 1.0}
    return float(np.mean([severity[entry["psi_level"]] for entry in result.values()]))
