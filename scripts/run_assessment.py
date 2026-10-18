"""Run a full FairGuard assessment from the command line.

Example (after the demo data and models are generated):

    python scripts/run_assessment.py --sensitive gender
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
    parser.add_argument("--reference", default=str(config.DEMO_DATA), help="reference data for drift")
    parser.add_argument("--label", default="approved", help="label column")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--texts", default=None, help="optional CSV with a 'text' column (and 'reference')")
    parser.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    args = parser.parse_args()

    df = data.load_csv(args.data)
    artifact = train.load_model(args.model)
    reference = data.load_csv(args.reference) if args.reference else None

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
    )

    paths = report.save_report(assessment)
    print(f"\nRisk score: {assessment['risk']['score']} -> {assessment['risk']['level']} "
          f"(suggested decision: {assessment['decision']})")
    print("\nRecommendations:")
    for line in assessment["recommendations"]:
        print(f"  - {line}")
    print("\nSaved:")
    for kind, path in paths.items():
        print(f"  {kind}: {path}")


if __name__ == "__main__":
    main()
