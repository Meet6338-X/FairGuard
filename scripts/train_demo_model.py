"""Train baseline models on the demo dataset or any real credit-style CSV.

Run from the repo root:

    python scripts/train_demo_model.py                                   # demo data
    python scripts/train_demo_model.py --data data/adult.csv \
        --label income_gt_50k --sensitive sex --model-type logistic_regression

With --data, the sensitive attribute is kept out of the model features (the
usual fair-AI setup); fairness is still measured on it at assessment time.
Labels with two values and numeric/string features are both fine.
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

    if len(pd.unique(df[args.label if args.data else train.LABEL].dropna())) != 2:
        raise SystemExit("label column must have exactly two distinct values")

    raw = features if args.data else train.FEATURES
    for model_type in model_types:
        model, X_train, X_test, y_train, y_test = train.train_model(
            df, model_type, label=args.label if args.data else train.LABEL, features=features
        )
        name = f"{model_type}.pkl" if args.data is None else f"{stem}_{model_type}.pkl"
        path = train.save_model(model, config.MODELS_DIR / name, X_train.columns, model_type, raw_features=raw)
        accuracy = (model.predict(X_test) == y_test).mean()
        print(f"{model_type:20s} test accuracy={accuracy:.3f}  -> {path}")
    print("tip: run scripts/run_assessment.py next")


if __name__ == "__main__":
    main()
