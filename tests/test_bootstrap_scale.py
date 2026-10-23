"""RC-24 regression — the bootstrap must iterate over *groups*, not rows.

The first version of `_per_group_rates` wrote ``for group in groups:``, which
walks every row of the array and re-scans the whole vector with
``groups == group`` for each of them.  The bootstrap therefore cost
O(n_boot * rows^2): the demo's 500 rows still finished in a couple of
seconds, but a real 12 000-row hold-out spent 10+ minutes per assessment.
"""
import time

import numpy as np
import pandas as pd

from fairguard import fairness


def _big_groups(n=12000):
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, n)
    y_pred = rng.integers(0, 2, n)
    sensitive = pd.Series(rng.choice(["M", "F"], n))
    return y_true, y_pred, sensitive


def test_per_group_rates_walks_unique_groups():
    y_true = np.array([1, 1, 1, 0])
    y_pred = np.array([1, 1, 0, 0])
    groups = np.array(["A", "A", "B", "B"])
    rates = fairness._per_group_rates(y_true, y_pred, groups)
    # Walk-every-row would smear the groups together; per group:
    # A: y_true=[1,1], y_pred=[1,1] -> accuracy 1.0, selection 1.0
    # B: y_true=[1,0], y_pred=[0,0] -> one miss, one correct TN: 0.5 / 0.0.
    assert len(rates) == 2
    assert rates["A"]["accuracy"] == 1.0
    assert rates["B"]["accuracy"] == 0.5
    assert rates["A"]["selection_rate"] == 1.0
    assert rates["B"]["selection_rate"] == 0.0


def test_per_group_rates_works_on_pandas_and_numpy_arrays():
    y_true, y_pred, sensitive = _big_groups(500)
    from_series = fairness._per_group_rates(y_true, y_pred, sensitive)
    from_numpy = fairness._per_group_rates(y_true, y_pred, sensitive.to_numpy())
    assert from_series["M"]["n"] == from_numpy["M"]["n"]
    assert from_series["F"]["n"] == from_numpy["F"]["n"]


def test_bootstrap_is_fast_on_twelve_thousand_rows():
    y_true, y_pred, sensitive = _big_groups()
    start = time.perf_counter()
    result = fairness.bootstrap_intervals(y_true, y_pred, sensitive)
    elapsed = time.perf_counter() - start
    assert result, "the bootstrap should run on data of this size"
    assert result["n_boot"] == 200
    assert elapsed < 5.0, f"bootstrap took {elapsed:.1f}s — RC-24 regression?"
