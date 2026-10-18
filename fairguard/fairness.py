"""Fairness assessment with Fairlearn — group metrics plus disparity measures."""
from functools import partial

import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score

from . import config


def assess_fairness(y_true, y_pred, sensitive: pd.Series, sensitive_name="attribute") -> dict:
    """Group-wise metrics and the two headline disparity measures."""
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
    return {
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
        "group_sizes": {str(k): int(v) for k, v in pd.Series(sensitive).value_counts().items()},
    }


def fairness_risk(result: dict, dpd_limit=None, eod_limit=None) -> float:
    """Worst disparity mapped to 0..1 (0.5 at the limit, 1.0 at twice the limit)."""
    dpd_limit = dpd_limit or config.FAIRNESS_DPD_LIMIT
    eod_limit = eod_limit or config.FAIRNESS_EOD_LIMIT
    worst = max(
        result["demographic_parity_difference"] / dpd_limit,
        result["equalized_odds_difference"] / eod_limit,
    )
    return float(min(worst / 2, 1.0))
