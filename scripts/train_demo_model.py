"""Train baseline models on the demo dataset or any real credit-style CSV.

Run from the repo root:

    python scripts/train_demo_model.py                                   # demo data
    python scripts/train_demo_model.py --data data/adult.csv \
        --label income_gt_50k --sensitive sex --model-type logistic_regression

With --data, the sensitive attribute is kept out of the model features (the
usual fair-AI setup); fairness is still measured on it at assessment time.
Labels with two values and numeric/string features are both fine.

Every model artifact also stores the **hold-out split** it was trained with
(RC-02).  `scripts/run_assessment.py` then scores the model on those unseen
rows, so the reported accuracy is out-of-sample instead of in-sample.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `import fairguard`
from fairguard import config, data, train


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=None, help="CSV file (default: demo loans data)")
    parser.add_argument("--label", default=train.LABEL, help="target column in --data")
    parser.add_argument("--sensitive", default=None, help="sensitive column to keep out of the model")
    parser.add_argument("--model-type", choices=train.MODEL_TYPES + ["all", "demo"], default=None,
                        help="one model type, 'all' (both, real data) or 'demo' (both, demo data)")
    parser.add_argument("--test-size", type=float, default=config.TEST_SPLIT,
                        help="share of rows kept aside for the hold-out evaluation")
    parser.add_argument("--no-balance", action="store_true",
                        help="disable balanced class weights (rarely a good idea)")
    args = parser.parse_args()

    if args.data:
        df = data.load_csv(args.data)
        if args.label not in df.columns:
            raise SystemExit(f"label '{args.label}' not in columns: {sorted(df.columns)}")
        features = train.pick_features(df, args.label, args.sensitive)
        model_types = train.MODEL_TYPES if args.model_type in (None, "all") else [args.model_type]
        stem = Path(args.data).stem
    else:
        df = data.load_csv(config.DEMO_DATA)
        features = None  # demo FEATURES (gender kept in on purpose)
        model_types = train.MODEL_TYPES if args.model_type in (None, "all", "demo") else [args.model_type]
        stem = "demo"

    label = args.label if args.data else train.LABEL
    if not data.is_binary_label(df[label]):
        uniques = list(pd.unique(df[label].dropna()))[:5]
        raise SystemExit(
            f"label column '{label}' must have at most two distinct values, found {uniques}"
        )

    raw = features if args.data else train.FEATURES
    balance = not args.no_balance
    for model_type in model_types:
        model, X_train, X_test, y_train, y_test = train.train_model(
            df, model_type, label=label, features=features,
            test_size=args.test_size, balance_classes=balance,
        )
        name = f"{model_type}.pkl" if args.data is None else f"{stem}_{model_type}.pkl"
        # RC-02: store the hold-out so the assessment can be scored out-of-sample.
        holdout = train.make_holdout(df, test_size=args.test_size, label=label, features=features)
        path = train.save_model(
            model, config.MODELS_DIR / name, X_train.columns, model_type,
            raw_features=raw, holdout=holdout,
            meta={"label": label, "balanced": balance,
                  "test_accuracy": float((model.predict(X_test) == y_test).mean())},
        )
        accuracy = (model.predict(X_test) == y_test).mean()
        print(f"{model_type:20s} hold-out accuracy={accuracy:.3f} "
              f"(n_test={holdout['n_test']}, stratified={holdout['stratified']})  -> {path}")
    print("tip: run scripts/run_assessment.py next")
    print("      the assessment scores the hold-out rows by default "
          "(use --full-dataset to compare in-sample numbers)")


if __name__ == "__main__":
    main()
