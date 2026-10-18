"""FairGuard AI — Streamlit dashboard.

Run from the repo root:

    streamlit run app.py

Flow: pick a dataset + model in the sidebar → run the assessment once →
explore the results in the tabs. Everything runs locally.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fairguard import config, data, llm_eval, pipeline, report, tracking, train  # noqa: E402
from scripts.make_demo_data import make  # noqa: E402

st.set_page_config(page_title="FairGuard AI", page_icon="🛡️", layout="wide")


# --- helpers ---------------------------------------------------------------

@st.cache_data(show_spinner=False)
def demo_dataframe() -> pd.DataFrame:
    """The built-in demo dataset; generated on the fly when no CSV exists yet."""
    if Path(config.DEMO_DATA).exists():
        return data.load_csv(config.DEMO_DATA)
    return make(n=1000)


def available_models() -> list[Path]:
    return sorted(Path(config.MODELS_DIR).glob("*.pkl"))


def run_pipeline(df, upload, sensitive, label, threshold, use_reference, texts):
    artifact = train.load_model(upload)
    text_result = None
    if texts:
        text_result = llm_eval.evaluate_texts(texts)

    # If the model's raw features aren't all present in the current dataset,
    # try to auto-load the dataset implied by the model filename
    # (e.g. adult_logistic_regression.pkl -> adult.csv).
    raw_features = artifact.get("raw_features") or []
    missing = [c for c in raw_features if c not in df.columns]
    if missing:
        implied = None
        stem = upload.stem  # e.g. "adult_logistic_regression"
        for model_type in train.MODEL_TYPES:
            suffix = f"_{model_type}"
            if stem.endswith(suffix):
                implied = stem[: -len(suffix)] + ".csv"
                break
        if implied:
            implied_path = config.DATA_DIR / implied
            if implied_path.exists():
                st.info(
                    f"This model was trained on **{implied}** — switching dataset "
                    f"from the current selection to match."
                )
                df = data.load_csv(implied_path)

    with st.spinner("Running the full assessment (fairness, SHAP, privacy, drift)…"):
        return pipeline.run_assessment(
            df=df,
            model=artifact["model"],
            features=artifact["features"],
            raw_features=artifact.get("raw_features"),
            label=label,
            sensitive=sensitive,
            reference=df if use_reference else None,
            model_id=upload.stem,
            model_version=artifact.get("model_type", "dev"),
            dataset_id=upload.stem,
            threshold=threshold,
            text=text_result,
            extra_artifacts=[str(upload)],
        )


def importance_frame(importance: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {"feature": list(importance), "mean |SHAP|": list(importance.values())}
    )


# --- sidebar ---------------------------------------------------------------

st.sidebar.title("🛡️ FairGuard AI")
st.sidebar.caption("Responsible-AI assessment workbench")

source = st.sidebar.radio("Dataset", ["Demo dataset", "Upload CSV"])
if source == "Demo dataset":
    df = demo_dataframe()
else:
    upload = st.sidebar.file_uploader("CSV file", type="csv")
    if upload is None:
        st.sidebar.info("Upload a CSV with a binary label column to continue.")
        st.stop()
    df = pd.read_csv(upload)

models = available_models()
if not models:
    st.sidebar.error("No trained models found. Run: python scripts/train_demo_model.py")
    st.stop()
model_path = st.sidebar.selectbox("Model", models, format_func=lambda p: p.stem)

columns = list(df.columns)
label = st.sidebar.selectbox(
    "Label column", columns, index=columns.index("approved") if "approved" in columns else 0
)
sensitive_default = "gender" if "gender" in columns else columns[0]
sensitive = st.sidebar.selectbox(
    "Sensitive attribute", columns, index=columns.index(sensitive_default)
)
threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)
use_reference = st.sidebar.checkbox("Use this dataset as drift reference", value=True)
texts_block = st.sidebar.text_area(
    "Optional: texts to check (one per line)",
    placeholder="Paste model outputs here to check toxicity / bias / PII…",
)

if st.sidebar.button("Run assessment", type="primary"):
    texts = [line for line in texts_block.splitlines() if line.strip()]
    try:
        st.session_state["assessment"] = run_pipeline(
            df, model_path, sensitive, label, threshold, use_reference, texts
        )
        st.session_state["explain_df"] = df
    except ValueError as exc:
        st.sidebar.error(f"Assessment failed: {exc}")
        st.stop()

st.sidebar.markdown("---")
st.sidebar.caption(
    "Local only · no paid APIs · fairness numbers describe the model on this "
    "dataset, not people."
)

# --- main ------------------------------------------------------------------

st.title("FairGuard AI — Responsible AI Assessment Workbench")
assessment = st.session_state.get("assessment")
if assessment is None:
    st.info("Configure the assessment in the sidebar, then press **Run assessment**.")
    st.stop()

tabs = st.tabs(
    ["Overview", "Fairness", "Explainability", "Privacy", "Drift",
     "Performance", "Text & LLM", "Review", "History", "Report"]
)

# Overview
with tabs[0]:
    left, mid, right = st.columns(3)
    left.metric("Risk score", f"{assessment['risk']['score']:.2f}")
    mid.metric("Risk level", assessment["risk"]["level"])
    right.metric("Suggested decision", assessment["decision"])
    contributions = pd.Series(assessment["risk"]["contributions"], name="contribution")
    st.plotly_chart(px.bar(contributions.sort_values(), orientation="h",
                           title="Risk contributions"), use_container_width=True)
    st.subheader("Recommendations")
    for item in assessment["recommendations"]:
        st.markdown(f"- {item}")
    st.subheader("Dataset profile")
    st.dataframe(pd.DataFrame(data.profile_dataset(df)["numeric_summary"]).T)

# Fairness
with tabs[1]:
    fairness = assessment["fairness"]
    c1, c2 = st.columns(2)
    c1.metric("Demographic parity difference",
              f"{fairness['demographic_parity_difference']:.3f}")
    c2.metric("Equalized odds difference",
              f"{fairness['equalized_odds_difference']:.3f}")
    st.caption(report.DISCLAIMER)
    metrics_order = ["selection_rate", "accuracy", "precision", "recall"]
    frame = pd.DataFrame(fairness["by_group"]).reindex(metrics_order)
    st.dataframe(frame)
    long = frame.reset_index().melt(id_vars="index", var_name="group", value_name="value")
    st.plotly_chart(
        px.bar(long, x="index", y="value", color="group", barmode="group",
               title="Metrics by group"),
        use_container_width=True,
    )
    st.caption(f"Group sizes: {fairness['group_sizes']}")

# Explainability
with tabs[2]:
    explainability = assessment.get("explainability")
    if not explainability:
        st.info("Explainability was disabled for this run.")
    else:
        st.plotly_chart(
            px.bar(importance_frame(explainability["global_importance"]).head(10),
                   x="mean |SHAP|", y="feature", orientation="h",
                   title="Global importance — top 10 features"),
            use_container_width=True,
        )
        st.subheader("Single prediction")
        explain_df = st.session_state["explain_df"]
        row = st.slider("Row", 0, len(explain_df) - 1, 0)
        from fairguard import explainability as expl  # local import keeps startup light

        artifact = train.load_model(model_path)
        X = train.encode_features(explain_df, features=artifact.get("raw_features"),
                                  columns=artifact["features"])
        local = expl.local_explanation(artifact["model"], X, row_index=row)
        st.write(f"Predicted approval probability: **{local['probability']:.2f}**")
        contrib = pd.Series(local["contributions"], name="SHAP value")
        st.plotly_chart(px.bar(contrib, title="Feature contributions (signed)"),
                        use_container_width=True)

# Privacy
with tabs[3]:
    privacy = assessment["privacy"]
    st.metric("Privacy risk", f"{privacy['risk']:.2f}")
    if privacy["findings"]:
        st.dataframe(pd.DataFrame(privacy["findings"]))
        st.caption("`samples` are masked on purpose — no raw PII is displayed.")
    else:
        st.success("No PII-like columns detected.")

# Drift
with tabs[4]:
    drift = assessment.get("drift")
    if not drift:
        st.info("Drift was not requested for this run (no reference dataset).")
    else:
        drift_frame = pd.DataFrame(drift).T.reset_index(names="feature")
        st.dataframe(drift_frame)
        colors = {"stable": "#2ecc71", "moderate": "#f39c12", "major": "#e74c3c"}
        st.plotly_chart(
            px.bar(drift_frame, x="feature", y="psi", color="psi_level",
                   color_discrete_map=colors, title="PSI per feature"),
            use_container_width=True,
        )

# Performance
with tabs[5]:
    performance = assessment["performance"]
    c = st.columns(5)
    for column, key in zip(c, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
        column.metric(key.upper(), f"{performance.get(key, float('nan')):.3f}")
    matrix = performance["confusion_matrix"]
    st.plotly_chart(
        px.imshow(matrix, text_auto=True, color_continuous_scale="Blues",
                  labels=dict(x="Predicted", y="Actual"),
                  x=["rejected", "approved"], y=["rejected", "approved"]),
        use_container_width=True,
    )

# Text & LLM
with tabs[6]:
    text = assessment.get("text")
    if not text:
        st.info("No texts were provided for this run.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Overall text risk", f"{text['risk']:.2f}")
        c2.metric("Flagged texts", f"{text['flagged']} / {text['n_texts']}")
        st.caption(f"Backend: {text['backend']} — install transformers+torch for local model judges.")
        st.plotly_chart(
            px.bar(pd.Series(text["kind_summary"], name="mean score"),
                   title="Mean score per check"),
            use_container_width=True,
        )
        st.dataframe(pd.DataFrame(text["per_text"]))

# Review
with tabs[7]:
    decisions = ["pending", "Approve", "Approve with monitoring", "Needs review", "Block"]
    current = assessment.get("reviewer_decision") or "pending"
    choice = st.selectbox("Reviewer decision", decisions,
                          index=decisions.index(current) if current in decisions else 0)
    note = st.text_input("Reviewer note", value=assessment.get("reviewer_note", ""))
    if st.button("Save decision to this assessment"):
        assessment["reviewer_decision"] = None if choice == "pending" else choice
        assessment["reviewer_note"] = note or None
        st.session_state["assessment"] = assessment
        st.success("Decision saved (it will appear in the exported report).")

# History
with tabs[8]:
    frame = tracking.history()
    if frame.empty:
        st.info("No MLflow runs yet — assessments appear here once logged.")
    else:
        st.dataframe(frame)

# Report
with tabs[9]:
    st.download_button("Download JSON", report.to_json(assessment), "assessment.json")
    st.download_button("Download Markdown report", report.build_markdown(assessment),
                       "assessment.md", "text/markdown")
    st.download_button("Download model card", report.build_model_card(assessment),
                       "model_card.md", "text/markdown")
    st.subheader("Markdown preview")
    st.markdown(report.build_markdown(assessment))
