"""Retrain every model in models/ with the current train.py.

Old artifacts in models/ were saved with earlier semantics: some predate the
hold-out split (RC-02), some predate one-hot encoded categoricals (RC-03),
some predate the leakage guard (RC-23, the adult `income` copy) and the
balanced class weights (RC-19).  This script rebuilds them all so every
artifact carries

* a stored hold-out split (out-of-sample assessment by default),
* balanced class weights,
* one-hot encoded low-cardinality categoricals,
* the label-duplicate (leakage) guard.

Run from the repo root:

    python scripts/retrain_all.py          # all datasets, all model types
    python scripts/retrain_all.py --only adult       # one dataset
    python scripts/retrain_all.py --models logistic_regression

The dataset/label/sensitive mapping below mirrors scripts/download_datasets.py
(same file names, same label columns, same准备好 sensitive attributes).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `import fairguard`
from fairguard import config, data, train

# dataset stem -> (label, sensitive). demo = the synthetic demo loans data.
DATASETS = {
    "demo":                ("approved",              "gender"),        # demo_loans.csv
    "adult":               ("income_gt_50k",         "sex"),
    "german":              ("credit_good",           "sex"),
    "taiwan_credit":       ("default_next_month",    "gender"),
    "compas_two_years":    ("two_year_recid",        "race"),
    "bank_marketing":      ("subscribed",            "age_group"),
    "student_performance": ("G3_pass",               "sex"),
    "heart":               ("disease",               "sex"),
    "communities":         ("high_violent_crime",    "majority_white"),
    "australian":          ("credit_approved",       "gender_like"),
}

DATA_FILES = {"demo": ["demo_loans.csv"]}


def data_file(dataset: str) -> Path:
    if dataset in DATA_FILES:
        return config.DATA_DIR / DATA_FILES[dataset][0]
    path = config.DATA_DIR / f"{dataset}.csv"
    return path if path.exists() else config.DATA_DIR / f"{dataset}_rb.csv"


def retrain(dataset: str, label: str, sensitive: str, model_types, balance=True) -> list[dict]:
    df = data.load_csv(data_file(dataset))
    if df is None or label not in df.columns:
        return [{"dataset": dataset, "error": f"label '{label}' or file not found"}]
    if not data.is_binary_label(df[label]):
        return [{"dataset": dataset, "error": "label is not binary"}]

    features = None if dataset == "demo" else train.pick_features(df, label, sensitive)
    out = []
    for model_type in model_types:
        name = f"{dataset}_{model_type}.pkl"
        path = config.MODELS_DIR / name
        try:
            model, X_train, X_test, y_train, y_test = train.train_model(
                df, model_type, label=label, features=features,
                balance_classes=balance,
            )
            # identical args -> the hold-out stored in the artifact matches the
            # split train_model used, so assessments are truly out-of-sample
            holdout = train.make_holdout(df, label=label, features=features)
            test_accuracy = float((model.predict(X_test) == y_test).mean())
            train.save_model(
                model, path, X_train.columns, model_type,
                raw_features=features if features is not None else train.FEATURES,
                holdout=holdout,
                meta={"label": label, "balanced": balance, "test_accuracy": test_accuracy},
            )
            out.append({"dataset": dataset, "model": name, "accuracy": round(test_accuracy, 4),
                        "n_test": holdout["n_test"], "features": len(features or train.FEATURES)})
        except Exception as exc:  # noqa: BLE001 - report per model, keep going
            out.append({"dataset": dataset, "model": name, "error": str(exc)})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", default=None, help="retrain one dataset (stem name)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="restrict model types (default: all three)")
    parser.add_argument("--no-balance", action="store_true")
    args = parser.parse_args()

    model_types = args.models or list(train.MODEL_TYPES)
    balance = not args.no_balance
    model_types = [m for m in model_types if m in train.MODEL_TYPES] or list(train.MODEL_TYPES)

    rows = []
    for dataset, (label, sensitive) in DATASETS.items():
        if args.only and dataset != args.only:
            continue
        rows.extend(retrain(dataset, label, sensitive, model_types, balance=balance))

    frame = pd.DataFrame(rows)
    if frame.empty:
        print("nothing to do")
        return
    for column in ("accuracy", "n_test", "features", "error"):
        if column not in frame.columns:
            frame[column] = None
    frame = frame[["dataset", "model", "accuracy", "n_test", "features", "error"]]
    print(frame.to_string(index=False))
    failures = frame[frame["error"].notna()]
    if len(failures):
        print(f"\n{len(failures)} model(s) failed — see the error column above.")
    print(f"\ndone: {len(frame)} artifacts refreshed into {config.MODELS_DIR}")


if __name__ == "__main__":
    main()
