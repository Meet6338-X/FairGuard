"""Fairness assessment with Fairlearn — group metrics plus disparity measures.

Root causes fixed here
-----------------------
* **RC-11** `MetricFrame` refuses to build with missing values in the
  sensitive attribute (``ValueError: Feature 'gender' contains missing
  values``).  Rows with an unknown group are now dropped and the number of
  dropped rows is reported.
* **RC-18** Fairness numbers used to be bare point estimates.  A
  demographic-parity difference of 0.30 on 30 rows is noise, not a finding.
  The disparity measures now carry bootstrap confidence intervals and groups
  that are too small to judge are flagged explicitly.
"""
from functools import partial

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score

from . import config


def _safe_divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def _per_group_rates(y_true, y_pred, groups):
    """Per-group selection rate, accuracy, precision, recall, TPR and FPR."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    groups = np.asarray(groups)
    unique = np.unique(groups)          # RC-24: was "for group in groups" —
    out = {}                            # i.e. one full pass *per row* instead
    for group in unique:                # of per group.  O(n) -> O(n * groups).
        mask = groups == group
        yt, yp = y_true[mask], y_pred[mask]
        positives = yt == 1
        negatives = yt == 0
        tp = float(np.sum((yp == 1) & positives))
        fp = float(np.sum((yp == 1) & negatives))
        fn = float(np.sum((yp == 0) & positives))
        tn = float(np.sum((yp == 0) & negatives))
        out[group] = {
            "selection_rate": _safe_divide(np.sum(yp), len(yp)),
            "accuracy": _safe_divide(np.sum(yp == yt), len(yp)),
            "precision": _safe_divide(tp, tp + fp),
            "recall": _safe_divide(tp, tp + fn),
            "true_positive_rate": _safe_divide(tp, tp + fn),
            "false_positive_rate": _safe_divide(fp, fp + tn),
            "n": int(len(yp)),
        }
    return out


def _disparities(rates):
    """Demographic-parity and equalized-odds differences from per-group rates."""
    if not rates:
        return 0.0, 0.0
    selection = [r["selection_rate"] for r in rates.values()]
    tpr = [r["true_positive_rate"] for r in rates.values()]
    fpr = [r["false_positive_rate"] for r in rates.values()]
    dpd = max(selection) - min(selection)
    eod = max(max(tpr) - min(tpr), max(fpr) - min(fpr))
    return float(dpd), float(eod)


def bootstrap_intervals(y_true, y_pred, sensitive, n_boot=None,
                         quantiles=None, random_state=None):
    """Stratified bootstrap confidence intervals for the disparity measures.

    Rows are resampled *within* each sensitive group, so every resample keeps
    all groups represented and no resample is discarded.
    """
    n_boot = n_boot if n_boot is not None else config.FAIRNESS_BOOTSTRAP
    quantiles = quantiles or config.FAIRNESS_CI_QUANTILES
    random_state = config.RANDOM_STATE if random_state is None else random_state

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    groups = np.asarray(sensitive)

    unique_groups = pd.unique(groups)
    if len(unique_groups) < 2 or n_boot <= 0:
        return {}

    indices_by_group = {g: np.flatnonzero(groups == g) for g in unique_groups}
    # A bootstrap interval from a handful of rows is noise, not information.
    # Below the thresholds the groups are reported as "small" instead and the
    # risk score keeps using the plain point estimate.
    if min(len(idx) for idx in indices_by_group.values()) < config.FAIRNESS_MIN_GROUP_ROWS:
        return {}
    if len(groups) < config.FAIRNESS_MIN_BOOTSTRAP_ROWS:
        return {}

    indices_by_group = {g: np.flatnonzero(groups == g) for g in unique_groups}
    rng = np.random.default_rng(random_state)
    dpd_samples, eod_samples = [], []
    for _ in range(n_boot):
        picks = np.concatenate([
            rng.choice(idx, size=len(idx), replace=True) for idx in indices_by_group.values()
        ])
        rates = _per_group_rates(y_true[picks], y_pred[picks], groups[picks])
        dpd, eod = _disparities(rates)
        dpd_samples.append(dpd)
        eod_samples.append(eod)

    low, high = quantiles
    return {
        "demographic_parity_difference_ci": [
            round(float(np.quantile(dpd_samples, low)), 4),
            round(float(np.quantile(dpd_samples, high)), 4),
        ],
        "equalized_odds_difference_ci": [
            round(float(np.quantile(eod_samples, low)), 4),
            round(float(np.quantile(eod_samples, high)), 4),
        ],
        "n_boot": int(n_boot),
        "ci_level": int(round((high - low) * 100)),
    }


def assess_fairness(y_true, y_pred, sensitive: pd.Series, sensitive_name="attribute",
                    n_boot=None) -> dict:
    """Group-wise metrics and the two headline disparity measures."""
    sensitive = pd.Series(sensitive).reset_index(drop=True)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # RC-11: rows with an unknown sensitive value cannot belong to a group.
    known = sensitive.notna().to_numpy()
    dropped = int((~known).sum())
    if dropped:
        sensitive = sensitive[known].reset_index(drop=True)
        y_true = y_true[known]
        y_pred = y_pred[known]

    if len(sensitive) == 0:
        return {
            "sensitive_attribute": sensitive_name,
            "by_group": {},
            "demographic_parity_difference": 0.0,
            "equalized_odds_difference": 0.0,
            "group_sizes": {},
            "dropped_missing_groups": dropped,
            "small_groups": [],
        }

    frame = MetricFrame(
        metrics={
            "selection_rate": lambda yt, yp: yp.mean(),
            "accuracy": accuracy_score,
            "precision": partial(precision_score, zero_division=0),
            "recall": partial(recall_score, zero_division=0),
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,
    )
    result = {
        "sensitive_attribute": sensitive_name,
        "by_group": {
            metric: {str(group): round(float(value), 4) for group, value in frame.by_group[metric].items()}
            for metric in frame.by_group.columns
        },
        "demographic_parity_difference": round(
            float(demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive)), 4
        ),
        "equalized_odds_difference": round(
            float(equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive)), 4
        ),
        "group_sizes": {str(k): int(v) for k, v in sensitive.value_counts().items()},
    }
    if dropped:
        result["dropped_missing_groups"] = dropped
    result.update(bootstrap_intervals(y_true, y_pred, sensitive, n_boot=n_boot))

    # RC-18: groups that are too small for their numbers to mean much.
    result["small_groups"] = sorted(
        str(group) for group, size in result["group_sizes"].items()
        if size < config.MIN_GROUP_SIZE
    )
    return result


def _effective_disparity(point: float, ci, limit: float) -> float:
    """How large a disparity to score, given its confidence interval.

    * CI entirely below the limit  → the difference is not a violation.
    * CI entirely above the limit  → it is a violation; score the measured size.
    * CI straddles the limit       → uncertain; score the middle of the
      interval so thin data is neither waved away nor overstated.
    """
    if not ci:
        return float(point)
    low, high = float(ci[0]), float(ci[1])
    if high <= limit:
        return float(point)
    if low >= limit:
        return float(point)
    return (low + high) / 2.0


def fairness_risk(result: dict, dpd_limit=None, eod_limit=None) -> float:
    """Worst disparity mapped to 0..1 (0.5 at the limit, 1.0 at twice the limit).

    When confidence intervals are available they decide how confident the
    scoring can be: a difference that cannot be distinguished from the limit
    is scored from the middle of its interval instead of from a bare point
    estimate.
    """
    dpd_limit = dpd_limit or config.FAIRNESS_DPD_LIMIT
    eod_limit = eod_limit or config.FAIRNESS_EOD_LIMIT
    dpd = _effective_disparity(
        result.get("demographic_parity_difference", 0.0),
        result.get("demographic_parity_difference_ci"), dpd_limit)
    eod = _effective_disparity(
        result.get("equalized_odds_difference", 0.0),
        result.get("equalized_odds_difference_ci"), eod_limit)
    worst = max(dpd / dpd_limit, eod / eod_limit)
    return float(min(worst / 2, 1.0))
