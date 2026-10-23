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
# Hold-out share kept aside at training time so assessments can be scored
# out-of-sample instead of on rows the model was fitted on.
EVAL_SPLIT = 0.25

# Demo dataset files created by scripts/make_demo_data.py
DEMO_DATA = DATA_DIR / "demo_loans.csv"
DEMO_DATA_NEW = DATA_DIR / "demo_loans_new.csv"

# --- Fairness thresholds (disparity differences) ---
FAIRNESS_DPD_LIMIT = 0.10   # demographic parity difference
FAIRNESS_EOD_LIMIT = 0.10   # equalized odds difference
# Groups with fewer rows than this are reported but flagged as unreliable.
MIN_GROUP_SIZE = 30
# Bootstrap resamples used for the fairness confidence intervals.
FAIRNESS_BOOTSTRAP = 200
FAIRNESS_CI_QUANTILES = (0.025, 0.975)
# A bootstrap interval is only computed when every group has at least this
# many rows (and the sample has at least FAIRNESS_MIN_BOOTSTRAP_ROWS rows).
# Below that the interval is noise, and the groups are reported as small
# instead of driving the risk score.
FAIRNESS_MIN_GROUP_ROWS = 10
FAIRNESS_MIN_BOOTSTRAP_ROWS = 50

# --- Drift thresholds (SRS: PSI and KS test) ---
PSI_MODERATE = 0.10         # 0.10 – 0.25 = moderate shift
PSI_MAJOR = 0.25            # above = major shift
KS_ALPHA = 0.05             # KS p-value below this means the distributions differ
# How much of the drift risk is driven by feature importance (the rest by the
# plain average). 1.0 = only important features count, 0.0 = flat average.
DRIFT_IMPORTANCE_WEIGHT = 0.7

# --- Performance floor ---
MIN_ACCURACY = 0.70
MIN_F1 = 0.60
# Number of bins used for the expected-calibration-error calculation.
CALIBRATION_BINS = 10

# --- Privacy ---
PII_SAMPLE_ROWS = 3         # sample values kept per privacy finding (masked)
# Column-name hints that justify a PII pattern match even on a numeric column.
PII_NAME_TRIGGERS = {
    "credit_card": ("card", "ccnum", "cc_number", "pan"),
    "phone": ("phone", "mobile", "msisdn", "tel"),
    "aadhaar": ("aadhaar", "aadhar", "uidai"),
    "ip_address": ("ip", "ipaddr"),
}

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
# HF task per judge kind. `text-classification` models take a single string,
# `token-classification` models return per-token spans and `pair` models need
# a second text (the reference) — mixing these up used to make the optional
# transformer backend crash on the PII and hallucination judges.
JUDGE_TASKS = {
    "toxicity": "text-classification",
    "bias": "text-classification",
    "hate_speech": "text-classification",
    "hallucination": "pair",
    "pii": "token-classification",
}
TARGET_MODELS = {
    "distilgpt2": "distilgpt2",
    "flan-t5-base": "google/flan-t5-base",
}
