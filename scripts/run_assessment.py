"""Run a full FairGuard assessment from the command line.

Examples (run from the repo root):

    # demo: train first, then assess on the hold-out split
    python scripts/make_demo_data.py
    python scripts/train_demo_model.py
    python scripts/run_assessment.py --sensitive gender

    # a real dataset, scored out-of-sample
    python scripts/train_demo_model.py --data data/adult.csv \
        --label income_gt_50k --sensitive sex
    python scripts/run_assessment.py --data data/adult.csv \
        --model models/adult_logistic_regression.pkl \
        --label income_gt_50k --sensitive sex

    # drift against a *different* (earlier) reference dataset
    python scripts/run_assessment.py --reference data/adult_reference.csv

Evaluation scope
----------------
By default the model is scored on the **hold-out split** stored in the model
artifact (rows the model never saw, RC-02).  Pass ``--full-dataset`` to score
it on everything instead — the report will then say the numbers are in-sample.

Drift
-----
Drift is only computed when ``--reference`` points at a *different* dataset.
Pointing it at the assessed dataset makes every PSI exactly 0 by
construction, which looks like "no drift" but means nothing (RC-07).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `import fairguard`
from fairguard import config, data, llm_eval, pipeline, report, train


def main():
    parser = argparse.ArgumentParser(description="FairGuard AI — run a full assessment")
    parser.add_argument("--data", default=str(config.DEMO_DATA), help="dataset to assess (CSV)")
    parser.add_argument("--model", default=str(config.MODELS_DIR / "logistic_regression.pkl"))
    parser.add_argument("--sensitive", default="gender", help="sensitive attribute column")
    parser.add_argument("--reference", default=None,
                        help="reference dataset for drift (must be a DIFFERENT dataset; "
                             "omitting it skips the drift check)")
    parser.add_argument("--label", default="approved", help="label column")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--texts", default=None, help="optional CSV with a 'text' column (and 'reference')")
    parser.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    parser.add_argument("--full-dataset", action="store_true",
                        help="score the whole dataset (in-sample) instead of the hold-out split")
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="skip the bootstrap confidence intervals for the fairness metrics")
    args = parser.parse_args()

    df = data.load_csv(args.data)
    artifact = train.load_model(args.model)
    reference = data.load_csv(args.reference) if args.reference else None
    if reference is not None and len(reference) == len(df) and reference.equals(df):
        print("warning: --reference is the same dataset as --data — every PSI would be 0 "
              "by construction. The report marks this as a baseline, not as stability.",
              file=sys.stderr)

    # RC-02: score the hold-out rows when the artifact knows them.
    eval_df = None
    holdout = artifact.get("holdout")
    if not args.full_dataset and holdout:
        eval_df = train.holdout_rows(df, holdout)
        if len(eval_df) == len(df):
            eval_df = None  # no usable split -> fall back to the full dataset

    text_result = None
    if args.texts:
        texts = data.load_csv(args.texts)
        references = texts["reference"].tolist() if "reference" in texts.columns else None
        text_result = llm_eval.evaluate_texts(texts["text"].dropna().tolist(), references=references)

    assessment = pipeline.run_assessment(
        df=df,
        model=artifact["model"],
        features=artifact["features"],
        raw_features=artifact.get("raw_features"),
        label=args.label,
        sensitive=args.sensitive,
        reference=reference,
        model_id=Path(args.model).stem,
        model_version=artifact.get("model_type", "dev"),
        dataset_id=Path(args.data).stem,
        threshold=args.threshold,
        run_mlflow=not args.no_mlflow,
        extra_artifacts=[args.model],
        text=text_result,
        eval_df=eval_df,
        n_boot=0 if args.no_bootstrap else None,
    )

    paths = report.save_report(assessment)
    scope = assessment["evaluation"]
    print(f"\nEvaluation scope: {scope['scope']} ({scope['n_rows']} rows scored)")
    print(f"Risk score: {assessment['risk']['score']} -> {assessment['risk']['level']} "
          f"(suggested decision: {assessment['decision']})")
    print(f"Risk coverage: {assessment['risk'].get('coverage', 0):.0%} "
          f"(dimensions: {', '.join(assessment['risk'].get('dimensions_assessed') or [])})")
    print("\nRecommendations:")
    for line in assessment["recommendations"]:
        print(f"  - {line}")
    print("\nSaved:")
    for kind, path in paths.items():
        print(f"  {kind}: {path}")


if __name__ == "__main__":
    main()
