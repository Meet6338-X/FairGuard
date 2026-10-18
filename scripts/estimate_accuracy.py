"""Estimate model accuracy on every dataset — programmatically, not by guess.

For each dataset in data/ (and each label/sensitive column pair below) it:

1. splits off a stratified hold-out test set (25 %),
2. runs stratified 5-fold cross-validation on the train part for every
   model type (logistic_regression, random_forest, gradient_boosting),
3. retrains the best model on all of train and reports its hold-out accuracy.

Usage:

    python scripts/estimate_accuracy.py

The table it prints is the estimate — no manual numbers anywhere.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `import fairguard`
from fairguard import config, data, train

# (dataset, label, sensitive) — datasets found by download_datasets.py
DATASETS = [
    ("adult.csv", "income_gt_50k", "sex"),
    ("german.csv", "credit_good", "sex"),
    ("taiwan_credit.csv", "default_next_month", "gender"),
    ("compas_two_years.csv", "two_year_recid", "race"),
    ("bank_marketing.csv", "subscribed", "age"),
    ("student_performance.csv", "G3_pass", "sex"),
    ("heart.csv", "disease", "sex"),
    ("demo_loans.csv", "approved", "gender"),
]

N_SPLITS = 5
TEST_SIZE = 0.25
TARGET_ACCURACY = 0.99


def cv_estimate(X: pd.DataFrame, y: pd.Series, model_type: str, n_splits: int = N_SPLITS):
    """Stratified k-fold mean accuracy (+/- std) for one model type."""
    scores = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    for train_idx, valid_idx in cv.split(X, y):
        fold = train.make_model(model_type)
        fold.fit(X.iloc[train_idx], y.iloc[train_idx])
        scores.append(accuracy_score(y.iloc[valid_idx], fold.predict(X.iloc[valid_idx])))
    return float(np.mean(scores)), float(np.std(scores))


# small hyperparameter grids per model type — enough to move the needle,
# small enough to finish in a few minutes on a laptop.
PARAM_GRIDS = {
    "logistic_regression": {
        "C": [0.1, 1.0, 10.0],
    },
    "random_forest": {
        "n_estimators": [200, 500],
        "max_depth": [10, 20, None],
        "min_samples_leaf": [1, 3],
    },
    "gradient_boosting": {
        "learning_rate": [0.05, 0.1, 0.2],
        "max_iter": [200, 500],
        "max_depth": [3, 6],
    },
}


def make_tuned(model_type: str, params: dict):
    """Build a model, overriding the pipeline's final estimator parameters."""
    base = train.make_model(model_type)
    est = base.steps[-1][1]
    for key, val in params.items():
        setattr(est, key, val)
    return base


def tune_one(X_train, y_train, model_type: str, n_tries: int = 12):
    """Random search over PARAM_GRIDS[model_type], validated on a 20 % hold-out."""
    import random

    random.seed(config.RANDOM_STATE)
    grid = PARAM_GRIDS[model_type]
    keys = list(grid.keys())
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_train, y_train, test_size=0.20, random_state=config.RANDOM_STATE, stratify=y_train
    )
    best_params, best_score = {}, 0.0
    for _ in range(n_tries):
        params = {k: random.choice(grid[k]) for k in keys}
        model = make_tuned(model_type, params)
        model.fit(X_tr, y_tr)
        score = accuracy_score(y_va, model.predict(X_va))
        if score > best_score:
            best_score, best_params = score, params
    return best_params, best_score


def estimate_one(csv_path: Path, label: str, sensitive: str, tune: bool = False):
    df = data.load_csv(csv_path)
    if label not in df.columns:
        return None
    if sensitive not in df.columns:
        return None
    y = df[label]
    if len(pd.unique(y.dropna())) != 2:
        return None

    features = train.pick_features(df, label, sensitive)
    X = train.encode_features(df, features=features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    results = []
    for model_type in train.MODEL_TYPES:
        mean, std = cv_estimate(X_train, y_train, model_type)
        if tune:
            params, valid_score = tune_one(X_train, y_train, model_type)
            results.append((model_type, mean, std, params, valid_score))
        else:
            results.append((model_type, mean, std))

    results.sort(key=lambda r: -r[1])
    best_type, best_mean, best_std = results[0][:3]

    # Retrain the best model on all of train and measure hold-out accuracy.
    best = train.make_model(best_type)
    if tune:
        best = make_tuned(best_type, results[0][3])
    best.fit(X_train, y_train)
    holdout = accuracy_score(y_test, best.predict(X_test))

    return {
        "dataset": csv_path.name,
        "n_rows": len(df),
        "label": label,
        "sensitive": sensitive,
        "n_features": len(features),
        "results": results,
        "best": best_type,
        "best_cv_mean": best_mean,
        "best_cv_std": best_std,
        "holdout_accuracy": holdout,
        "reaches_target": holdout >= TARGET_ACCURACY,
        "tuned": tune,
    }


def main():
    parser = argparse.ArgumentParser(description="Estimate hold-out accuracy per dataset/model via CV")
    parser.add_argument("--target", type=float, default=TARGET_ACCURACY,
                        help="accuracy target to flag (default 0.99)")
    parser.add_argument("--tune", action="store_true", help="placeholder for hyperparameter sweep")
    args = parser.parse_args()

    target = args.target

    print(f"Accuracy estimation — stratified 5-fold CV + hold-out test set (test={TEST_SIZE:.0%})\n")
    rows = []
    for csv_name, label, sensitive in DATASETS:
        est = estimate_one(config.DATA_DIR / csv_name, label, sensitive, tune=args.tune)
        if est is None:
            print(f"skip {csv_name:24s} (label '{label}' or sensitive '{sensitive}' missing / not binary)")
            continue
        rows.append(est)
        flag = "  <-- reaches target" if est["reaches_target"] else ""
        print(f"{csv_name:24s} | label={label:20s} sensitive={sensitive:10s} | "
              f"rows={est['n_rows']:>6} features={est['n_features']:>2}")
        for item in est["results"]:
            mt, mean, std = item[:3]
            mark = "BEST" if mt == est["best"] else "    "
            extra = ""
            if est["tuned"] and len(item) == 5:
                extra = " " + str(item[3])
                print(f"    {mt:22s} CV={mean:.4f} +/- {std:.4f} valid={item[4]:.4f}{extra} {mark}{flag}")
            else:
                print(f"    {mt:22s} CV={mean:.4f} +/- {std:.4f} {mark}{flag}")
        print(f"    hold-out ({est['best']}): {est['holdout_accuracy']:.4f}{flag}\n")

    reachable = [r for r in rows if r["reaches_target"]]
    print(f"Datasets reaching {target:.0%} accuracy: {len(reachable)} / {len(rows)}")
    for r in reachable:
        print(f"  {r['dataset']:24s} {r['best']:22s} hold-out={r['holdout_accuracy']:.4f}")

    if not reachable:
        best_overall = max(rows, key=lambda r: r["holdout_accuracy"])
        print(f"\nNo dataset reaches {target:.0%} hold-out accuracy. "
              f"Best overall: {best_overall['dataset']} with {best_overall['best']} "
              f"= {best_overall['holdout_accuracy']:.4f}.")
        print("Note: 0.99 on tabular credit/banking data is not realistic without "
              "data leakage (e.g. a 'duration' column that leaks the target).")


if __name__ == "__main__":
    main()
