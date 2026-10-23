# FairGuard AI — Update log

> `UPDATE.md` records what shipped in each update round and what is planned next.
> The in-depth reasoning behind every root cause lives in
> [`ROOT_CAUSES.md`](ROOT_CAUSES.md); the "how to run it" lives in
> [`USER_MANUAL.md`](USER_MANUAL.md).

---

## Update 4 — 2026-10-23 — "Docs, diagrams, visualizations, everything measured"

### Added

* `docs/fairguard_architecture.svg` — vector architecture diagram (SVG) used as the README hero image.
* `visualize/` + `scripts/visualize_data.py` — 99 charts across the 11 datasets, PSI-drift between the two demo sets, and a text-safety panel; `visualize/README.md` explains how to read each chart and which red flags to look for (fairness signal, RC-23 leak, drift).
* Mermaid diagrams in the README ("How everything works"): assessment flowchart, risk-engine decision graph, API sequence diagram and the review-workflow state machine.
* Badges + team info in a professional README header.
* `UPDATE.md` (this file).
* `pytest.ini` — locks the pytest-asyncio fixture-loop scope and filters the starlette deprecation warning, so the test command exits cleanly on Windows.

### Fixed (quick recap — full list in ROOT_CAUSES.md)

* 24 root causes, each with the measurement that proved it and the test that pins it: wrong SHAP explainer (67× slower), train/test contamination, label leakage (`adult.csv` `income`), O(rows²) fairness bootstrap (10 min → 0.9 s), drift measured against itself, EOD ignored, ECE computed wrong, and more.
* All **34 model artifacts** retrained by `scripts/retrain_all.py` so each carries its hold-out split; assessments score out-of-sample by default.
* Assessment-time label-duplicate guard: a report with leakage says so at the top, and the record carries `assessment["leakage"]`.

### Status

* Test suite: **125 passed** (`python -m pytest tests -q`).
* End-to-end verified: CLI (demo + adult + australian), FastAPI, Streamlit, React build.
* Heuristic text judges active; transformer judges are wired but optional (plan below).

---

## Update 3 — 2026-10-22 — "Visualize section / dataset charts"

* `visualize/` suite (99 PNGs) and `scripts/visualize_data.py` shipped with the README + manual sections walking through every chart.

## Update 2 — 2026-10-22 — "Root-cause fixes land"

* RC-01…RC-20 fixes applied module by module, 59 → 113 tests, CLI/API/Streamlit smoke tests, USER_MANUAL.md + EXPLAIN.md + ROOT_CAUSES.md written.

## Update 1 — 2026-10-18 — "Initial workbench"

* 10 datasets, 3 model types, the six assessment dimensions, Streamlit + FastAPI + React front doors, MLflow tracking, CI on push.

---

# Planned: transformer text judges — status & plan

## Where we are today

The text-safety dimension (`text` in every assessment, `judges.py` + `llm_eval.py`) supports two kinds of backend:

| Backend | Available now | Used for | Speed | Notes |
|---|---|---|---|---|
| `heuristic` | ✅ always (offline) | every kind | instant | transparent word-list scoring (`judges.py::_word_risk`); shows its scores in every report tab |
| `transformer` | ⚠️ code is ready, **pending `pip install transformers torch`** | every kind | seconds | real model judges per kind |

The model **registry already exists** in `fairguard/config.py`:

```python
JUDGE_MODELS = {                      # HF model id per check
    "toxicity":       "unitary/toxic-bert",
    "bias":           "valurank/distilroberta-bias",
    "hate_speech":    "facebook/roberta-hate-speech-dynabench-r4-target",
    "hallucination":  "vectara/hallucination_evaluation_model",
    "pii":            "iiiorg/piiranha-v1-detect-personal-information",
}
JUDGE_TASKS = {                       # how each one must be loaded  (RC-12 fix)
    "toxicity": "text-classification",
    "bias":     "text-classification",
    "hate_speech": "text-classification",
    "hallucination": "pair",          # cross-encoder, reference required
    "pii":           "token-classification",
}
```

Two of the five kinds need **special wiring** because they are not plain classifiers — and the
code already handles that (the RC-12 fix):

* `hallucination` is a **pair** (cross-encoder): `(text, reference)` pairs from the `--texts` CSV's `reference` column.
* `pii` is a **token** classifier, scored by whether any PERSON token was found, not by the label probability.

## Why this matters

The heuristic backend is fast, explainable, offline. Machine judges catch what word lists miss:
"To call them animals" passes a word-list toxicity check only if "animals" is on the bad word list and a model gets it instantly.

## Plan (7 steps, ~1 hour of work each)

| Step | Detail | Acceptance check |
|---|---|---|
| **1. Install & pin** | `pip install transformers torch` then `requirements-ml.txt` with pins (`transformers>=4.40,<5`, `torch>=2.0`). `python -c "import transformers, torch"` | clean import, version in `requirements-ml.txt` |
| **2. Model registry sanity pass** | Run one `activate` smoke: for each kind, ensure the chosen model id loads on the *right* task per `JUDGE_TASKS`. Verify download sizes and license per model. Print the registry table. | **5× successfully created pipelines**; each download under 2 GB |
| **3. API contract smoke test** | Write a test that runs `evaluate_texts(data/sample_texts.csv["text"])` with `backend="transformer"` on the 5 kinds. Compare risk / flagged against the known heuristic baseline. | `backends` map is 5/5 `transformer`; PII finding on the card-number text, hallucination uses the `reference` |
| **4. Deterministic, cache-friendly settings** | `cache_dir` default, `low_cpu_mem`, manual_seed, no CLI-download-on-import in CI. Make sure CI (a free runner has no GPU or HF token) still passes: judge tests behind `pytest.importorskip("torch")` unless explicitly requested. | CI green without any transformers |
| **5. Backend comparison table in the docs** | **Experiments/evaluate**: heuristic vs transformer risk/flagged on the same 4 sample texts, written into USER_MANUAL "text / LLM" reference, allowing users to choose | a table with 5 kinds × 2 backends, measured factors, notes on when each is appropriate |
| **6. UI & API wiring** | Streamlit dropdown "text judge mode: heuristic / transformer (auto-fallback)"; when both are used show one row for each. The API already reports per-kind `backend`; add a "backend" filter to `/assess` | `backend` field per kind in the API response; no crash with either backend |
| **7. Document — "when to trust it"** | New EXPLAIN.md section: heuristic (fast, explainable, conservative); transformer (recommended when installed, calibrated per kind). Update README's Notes with a one-liner and UPDATE.md status | docs merged, honest caveat kept |

## Risks we are tracking

* Model download sizes (roBERTa-large ≈ 1.4 GB) — document, use `HfFolder` cache, never unload at import.
* Windows + MPI where `torch<2.2` on some Python builds — pin `torch>=2.2`.
* "vectara/hallucination_evaluation_model" outputs a "not-a-cross-encoder" shape if we skip the `pair` wiring — already handled, but pinned to test it.
* License on `iiiorg/piiranha-v1-detect-personal-information` — verify before redistributing.

## What's already done

* Per-kind HF **task** dispatch via `JUDGE_TASKS` (RC-12 fix) so `"token-classification"` for PII and `"pair"` for hallucination load correctly.
* The exact workbench contract: `llm_eval.evaluate_texts` accepts a `backend` argument; per-model **mis-Fallthrough** never silently happens (RC-14: `backends` shown per kind).
* The Streamlit / API / report surfaces all say which backend produced each kind's score.
