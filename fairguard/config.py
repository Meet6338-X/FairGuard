"""Central configuration: paths, thresholds, risk weights and judge models.

Everything a review meeting might want to tune lives here, in one place.
Values follow the SRS (PSI/KS drift thresholds, Fairlearn disparities) unless
noted otherwise.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MLRUNS_DIR = ROOT / "mlruns"

RANDOM_STATE = 42
TEST_SPLIT = 0.25

# Demo dataset files created by scripts/make_demo_data.py
DEMO_DATA = DATA_DIR / "demo_loans.csv"
DEMO_DATA_NEW = DATA_DIR / "demo_loans_new.csv"

# --- Fairness thresholds (disparity differences) ---
FAIRNESS_DPD_LIMIT = 0.10   # demographic parity difference
FAIRNESS_EOD_LIMIT = 0.10   # equalized odds difference

# --- Drift thresholds (SRS: PSI and KS test) ---
PSI_MODERATE = 0.10         # 0.10 – 0.25 = moderate shift
PSI_MAJOR = 0.25            # above = major shift
KS_ALPHA = 0.05             # KS p-value below this means the distributions differ

# --- Performance floor ---
MIN_ACCURACY = 0.70

# --- Privacy ---
PII_SAMPLE_ROWS = 3         # sample values kept per privacy finding (masked)

# --- Risk scoring: dimension weights, normalized over whatever was assessed ---
RISK_WEIGHTS = {
    "fairness": 0.30,
    "privacy": 0.20,
    "drift": 0.20,
    "performance": 0.15,
    "text": 0.15,
}
# Upper bounds for Low / Medium / High; above the last bound = Critical.
RISK_LEVEL_BOUNDS = ((0.25, "Low"), (0.50, "Medium"), (0.75, "High"))

# --- Text / LLM judge models (used when `transformers` is installed) ---
JUDGE_MODELS = {
    "toxicity": "unitary/toxic-bert",
    "bias": "valurank/distilroberta-bias",
    "hate_speech": "facebook/roberta-hate-speech-dynabench-r4-target",
    "hallucination": "vectara/hallucination_evaluation_model",
    "pii": "iiiorg/piiranha-v1-detect-personal-information",
}
TARGET_MODELS = {
    "distilgpt2": "distilgpt2",
    "flan-t5-base": "google/flan-t5-base",
}
