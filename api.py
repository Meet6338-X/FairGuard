"""FairGuard AI — FastAPI backend for the React frontend.

Run from the repo root:

    python api.py

Endpoints (all under /api):
    GET    /api/datasets        list CSV datasets in data/
    GET    /api/models          list trained .pkl models in models/
    POST   /api/train           train one or more models on a dataset
    POST   /api/assess          run a full responsible-AI assessment
    GET    /api/history         MLflow run history
    POST   /api/review          save a reviewer decision on an assessment
    GET    /api/report/{run_id}/{fmt}   json | md | card  — export a report
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))  # allow `import fairguard`
from fairguard import config, data, pipeline, report, tracking, train

app = FastAPI(title="FairGuard AI API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- request / response models ----------------------------------------------

class TrainRequest(BaseModel):
    dataset: Optional[str] = None          # filename in data/ (default: demo)
    label: str = "approved"
    sensitive: Optional[str] = None
    model_types: list[str] = ["logistic_regression", "random_forest", "gradient_boosting"]


class TrainResponse(BaseModel):
    results: list[dict]


class AssessRequest(BaseModel):
    dataset: str                           # filename in data/
    model: str                             # filename in models/
    label: str = "approved"
    sensitive: str = "gender"
    threshold: float = 0.5
    use_reference: bool = True             # assess drift against itself (no separate reference yet)
    texts: Optional[list[str]] = None
    model_id: Optional[str] = None
    model_version: str = "1.0"
    dataset_id: Optional[str] = None
    run_mlflow: bool = True


class ReviewRequest(BaseModel):
    run_id: str
    decision: str                          # pending | Approve | Approve with monitoring | Needs review | Block
    note: Optional[str] = None


# --- helpers ----------------------------------------------------------------

def _list_files(directory: Path, suffix: str) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    return sorted(p.name for p in directory.glob(f"*{suffix}"))


def _load_dataset(name: str) -> pd.DataFrame:
    path = config.DATA_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"dataset not found: {name}")
    return data.load_csv(path)


def _load_artifact(name: str):
    path = config.MODELS_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"model not found: {name}")
    return train.load_model(path)


def _assessment_id(assessment: dict) -> str:
    """Stable id matching the saved report filename stamp (YYYYMMDDTHHMMSS)."""
    ts = assessment.get("assessed_at", "")
    stamp = "".join(ch for ch in ts if ch.isalnum())[:15]
    return stamp or ts


# --- endpoints --------------------------------------------------------------

@app.get("/api/datasets")
def list_datasets():
    return {"datasets": _list_files(config.DATA_DIR, ".csv")}


@app.get("/api/models")
def list_models():
    return {"models": _list_files(config.MODELS_DIR, ".pkl")}


@app.post("/api/train", response_model=TrainResponse)
def train_models(req: TrainRequest):
    if req.dataset:
        df = _load_dataset(req.dataset)
        stem = Path(req.dataset).stem
    else:
        df = data.load_csv(config.DEMO_DATA)
        stem = "demo"
    if req.label not in df.columns:
        raise HTTPException(status_code=400, detail=f"label '{req.label}' not in dataset")
    if req.sensitive and req.sensitive not in df.columns:
        raise HTTPException(status_code=400, detail=f"sensitive '{req.sensitive}' not in dataset")

    features = train.pick_features(df, req.label, req.sensitive)
    out = []
    for model_type in req.model_types:
        model, X_train, X_test, y_train, y_test = train.train_model(
            df, model_type, label=req.label, features=features
        )
        name = f"{model_type}.pkl" if not req.dataset else f"{stem}_{model_type}.pkl"
        path = train.save_model(
            model, config.MODELS_DIR / name, X_train.columns, model_type,
            raw_features=features,
        )
        accuracy = (model.predict(X_test) == y_test).mean()
        out.append({
            "model": name, "model_type": model_type,
            "test_accuracy": round(float(accuracy), 4),
            "features": features,
        })
    return {"results": out}


@app.post("/api/assess")
def assess(req: AssessRequest):
    df = _load_dataset(req.dataset)
    artifact = _load_artifact(req.model)
    text_result = None
    if req.texts:
        from fairguard import llm_eval
        text_result = llm_eval.evaluate_texts(req.texts)

    reference = df if req.use_reference else None
    assessment = pipeline.run_assessment(
        df=df,
        model=artifact["model"],
        features=artifact["features"],
        raw_features=artifact.get("raw_features"),
        label=req.label,
        sensitive=req.sensitive,
        reference=reference,
        model_id=req.model_id or Path(req.model).stem,
        model_version=req.model_version,
        dataset_id=req.dataset_id or req.dataset,
        threshold=req.threshold,
        text=text_result,
        extra_artifacts=[str(config.MODELS_DIR / req.model)],
        run_mlflow=req.run_mlflow,
    )

    # persist JSON + markdown + model card
    saved = report.save_report(assessment, out_dir=config.REPORTS_DIR)
    return {
        "run_id": _assessment_id(assessment),
        "assessment": assessment,
        "saved": saved,
    }


@app.get("/api/history")
def history():
    frame = tracking.history()
    if frame.empty:
        return {"runs": []}
    return {"runs": json.loads(frame.to_json(orient="records"))}


@app.post("/api/review")
def review(req: ReviewRequest):
    decision_map = {"pending": None, "Approve": "Approve",
                    "Approve with monitoring": "Approve with monitoring",
                    "Needs review": "Needs review", "Block": "Block"}
    if req.decision not in decision_map:
        raise HTTPException(status_code=400, detail="invalid decision")
    # MLflow tags are immutable per run; record the decision as a new run param is
    # not supported retroactively, so we keep it in-memory for the dashboard and
    # store it on the latest matching run's assessment json instead.
    saved = list(config.REPORTS_DIR.glob("assessment_*.json"))
    if not saved:
        raise HTTPException(status_code=404, detail="no assessments found")
    latest = max(saved, key=lambda p: p.stat().st_mtime)
    assessment = json.loads(latest.read_text(encoding="utf-8"))
    assessment["reviewer_decision"] = decision_map[req.decision]
    assessment["reviewer_note"] = req.note
    latest.write_text(json.dumps(assessment, indent=2, default=str), encoding="utf-8")
    # also re-log to mlflow so history reflects it
    tracking.log_assessment(assessment, extra_artifacts=None)
    return {"ok": True, "run_id": latest.stem, "decision": assessment["reviewer_decision"]}


@app.get("/api/report/{run_id}/{fmt}")
def get_report(run_id: str, fmt: str):
    if fmt not in ("json", "md", "card"):
        raise HTTPException(status_code=400, detail="fmt must be json, md or card")
    matches = list(config.REPORTS_DIR.glob(f"assessment_{run_id}*.json")) or \
              list(config.REPORTS_DIR.glob(f"*_{run_id}*.json"))
    if not matches:
        # try newest file containing run_id substring
        matches = [p for p in config.REPORTS_DIR.glob("assessment_*.json")
                   if run_id in p.stem]
    if not matches:
        raise HTTPException(status_code=404, detail="report not found")
    assessment = json.loads(matches[0].read_text(encoding="utf-8"))
    if fmt == "json":
        return {"content": report.to_json(assessment), "filename": matches[0].name}
    if fmt == "md":
        return {"content": report.build_markdown(assessment), "filename": matches[0].name.replace(".json", ".md")}
    return {"content": report.build_model_card(assessment), "filename": matches[0].name.replace(".json", "_model_card.md")}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
