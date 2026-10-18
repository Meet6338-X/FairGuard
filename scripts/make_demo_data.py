"""Generate the synthetic demo dataset (biased lending decisions + PII columns).

Run from the repo root:

    python scripts/make_demo_data.py

Writes:
    data/demo_loans.csv      reference dataset
    data/demo_loans_new.csv  'after deployment' dataset with shifted distributions
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `import fairguard`
from fairguard import config

FIRST = ["Aarav", "Meera", "Rohan", "Isha", "Kabir", "Anaya", "Vikram", "Priya", "Dev", "Sara"]
LAST = ["Sharma", "Patel", "Kulkarni", "Desai", "Iyer", "Joshi", "Mehta", "Rao"]


def make(n: int = 2000, seed: int = 42, drift: bool = False) -> pd.DataFrame:
    """Synthetic loan applications with an unfair label, PII columns and optional drift.

    The label deliberately penalises women and older applicants at equal credit so
    the fairness module has something real to catch — this is a *teaching* dataset,
    not a statement about real applicants.
    """
    rng = np.random.default_rng(seed)
    gender = rng.choice(["male", "female"], size=n)
    region = rng.choice(["urban", "rural"], size=n, p=[0.7, 0.3])
    age = rng.integers(21, 65, size=n)
    income = rng.lognormal(mean=1.4, sigma=0.45, size=n)          # lakhs, median ≈ 4
    credit = np.clip(rng.normal(650, 90, size=n), 300, 900)
    employed_years = np.clip(rng.normal(8, 5, size=n), 0, 40)
    loans = rng.integers(0, 4, size=n)

    if drift:  # 'after deployment': incomes rise, credit scores slip
        income = income * 1.3
        credit = np.clip(credit - 60, 300, 900)

    logit = (
        2.0 * (
            -10.0
            + 0.02 * credit
            + 0.5 * income
            + 0.08 * employed_years
            - 0.15 * loans
            - 1.0 * (gender == "female")
            - 0.03 * np.clip(age - 45, 0, None)
        )
    )
    approved = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame({
        "applicant_id": [f"A{i:05d}" for i in range(n)],
        "age": age,
        "gender": gender,
        "region": region,
        "income_lakhs": income.round(2),
        "credit_score": credit.round(0).astype(int),
        "employed_years": employed_years.round(1),
        "existing_loans": loans,
        "approved": approved,
    })

    # PII-ish columns: only some rows carry contact details (realistic messiness)
    has_email = rng.random(n) < 0.7
    has_phone = rng.random(n) < 0.65
    df["email"] = [
        f"{f.lower()}.{l.lower()}{k}@example.com" if ok else None
        for f, l, ok, k in zip(rng.choice(FIRST, n), rng.choice(LAST, n), has_email, rng.integers(1, 99, n))
    ]
    df["phone"] = [
        f"+91{v}" if ok else None
        for ok, v in zip(has_phone, rng.integers(6_0000_00000, 9_9999_99999, n))
    ]
    df["full_name"] = [f"{f} {l}" for f, l in zip(rng.choice(FIRST, n), rng.choice(LAST, n))]
    return df


def main():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    reference = make()
    shifted = make(drift=True)
    reference.to_csv(config.DEMO_DATA, index=False)
    shifted.to_csv(config.DEMO_DATA_NEW, index=False)
    print(f"wrote {config.DEMO_DATA} ({len(reference)} rows)")
    print(f"wrote {config.DEMO_DATA_NEW} ({len(shifted)} rows) [drifted]")


if __name__ == "__main__":
    main()
