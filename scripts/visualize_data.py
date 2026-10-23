"""Generate the visualize/ folder — one suite of charts per dataset in data/.

Run from the repo root:

    python scripts/visualize_data.py                 # every dataset
    python scripts/visualize_data.py --only adult    # one dataset

For every dataset this writes, into visualize/<dataset>/:

    01_label_distribution.png        how balanced is the target
    02_numeric_distributions.png     histogram grid of the top numeric columns
    03_categorical_counts.png        bar grid of low-cardinality categorical columns
    04_label_correlation.png         signed correlation of every numeric column with the label
    05_correlation_heatmap.png       top-15 numeric columns (+ label) as a heatmap
    06_missing_values.png            % missing per column
    07_group_selection_rate.png      P(label=1) per sensitive group (+ group sizes)
    08_boxplots_top_features.png     box splits by label class for the top-6 numeric columns
    09_scatter_top_features.png      the two most label-correlated features, coloured by label

Plus two special charts:

    visualize/drift/10_drift_demo_loans.png     PSI per feature, demo_loans vs demo_loans_new
    visualize/text/11_text_safety.png           judge score per check for data/sample_texts.csv

Every chart is a PNG;  visualize/README.md is the browsing index.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless rendering, no window popping up
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fairguard import config, data, drift, llm_eval  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "visualize"

# label and sensitive column per dataset (same map as scripts/retrain_all.py)
DATASETS = {
    "demo_loans": ("approved", "gender"),
    "demo_loans_new": ("approved", "gender"),
    "adult": ("income_gt_50k", "sex"),
    "german": ("credit_good", "sex"),
    "taiwan_credit": ("default_next_month", "gender"),
    "compas_two_years": ("two_year_recid", "race"),
    "bank_marketing": ("subscribed", "age_group"),
    "student_performance": ("G3_pass", "sex"),
    "heart": ("disease", "sex"),
    "communities": ("high_violent_crime", "majority_white"),
    "australian": ("credit_approved", "gender_like"),
}
MAX_CATEGORICAL_CARDINALITY = 20
GRID_CAP = 16          # charts per grid (4 x 4)
TOP_N = 15             # features used by the correlation charts
GREEN, ORANGE, RED, BLUE, GREY = "#2ecc71", "#f39c12", "#e74c3c", "#3498db", "#95a5a6"
LEVEL_COLOR = {"stable": GREEN, "moderate": ORANGE, "major": RED}


def label_encoding(series: pd.Series):
    """-> ({0: neg_value, 1: pos_value}, series-encoded) or (None, None) if not binary."""
    values = series.dropna().unique()
    if len(values) != 2:
        return None, None
    mapping = data.binary_label_mapping(series)
    if not mapping:
        return None, None
    return mapping, series.map(mapping)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(include=[np.number]).columns)


def categorical_columns(df: pd.DataFrame) -> list[str]:
    keep = []
    for column in df.select_dtypes(include=["object", "bool", "category"]).columns:
        if df[column].nunique(dropna=True) <= MAX_CATEGORICAL_CARDINALITY:
            keep.append(column)
    return keep


def label_correlations(df: pd.DataFrame, label_series: pd.Series):
    """Signed Pearson correlation of every numeric column with the encoded label.

    The label column itself is excluded (it correlates 1.0 with itself and
    would push the real features out of the top ranks).
    """
    label_name = label_series.name
    encode, encoded = label_encoding(label_series)
    if encoded is None:
        return None
    numeric = [column for column in numeric_columns(df) if column != label_name]
    if not numeric:
        return None
    frame = df[numeric].copy()
    frame["_label"] = encoded.reindex(frame.index)
    correlations = frame.corr(numeric_only=True)["_label"].drop("_label", errors="ignore")
    return correlations.sort_values(key=np.abs, ascending=False)


def top_numeric_by_label(df: pd.DataFrame, label_series: pd.Series, n: int) -> list[str]:
    correlations = label_correlations(df, label_series)
    if correlations is not None and len(correlations) > 0:
        return list(correlations.index[:n])
    return numeric_columns(df)[:n]


def _grid(rows, columns, width=14.0, height_per_row=2.6):
    height = min(height_per_row * rows + 1.0, 20.0)
    figure, axes = plt.subplots(rows, columns, figsize=(width, height))
    return figure, np.atleast_1d(axes).ravel()


def _finish(figure, path: Path, title: str):
    figure.suptitle(title, fontsize=13, fontweight="bold")
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    figure.savefig(path, dpi=110)
    plt.close(figure)


# --- charts -----------------------------------------------------------------

def chart_label_distribution(df, label, out_dir):
    counts = df[label].value_counts(dropna=False)
    if len(counts) > 12:
        return
    figure, axis = plt.subplots(figsize=(7, 5))
    colors = [BLUE if value == counts.idxmax() else ORANGE for value in counts.index]
    bars = axis.bar([str(value) for value in counts.index], counts.values, color=colors)
    total = counts.sum()
    for bar, count in zip(bars, counts.values):
        axis.annotate(f"{count:,}\n({count / total:.0%})",
                      (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      ha="center", va="bottom", fontsize=10)
    axis.set_ylabel("rows")
    axis.set_title(f"Label '{label}' — {len(df):,} rows total", fontsize=12)
    axis.margins(y=0.18)
    figure.tight_layout()
    figure.savefig(out_dir / "01_label_distribution.png", dpi=110)
    plt.close(figure)


def chart_numeric_distributions(df, label, out_dir):
    columns = top_numeric_by_label(df, df[label], GRID_CAP)
    if not columns:
        return
    skipped = len(numeric_columns(df)) - len(columns)
    rows = math_rows(len(columns))
    figure, axes = _grid(rows, 4)
    for axis in axes:
        axis.set_visible(False)
    for axis, column in zip(axes, columns):
        axis.set_visible(True)
        axis.hist(df[column].dropna(), bins=30, color=BLUE, edgecolor="white", linewidth=0.4)
        axis.set_title(column, fontsize=10)
        axis.tick_params(labelsize=8)
    note = f" · {skipped} more columns not shown" if skipped > 0 else ""
    _finish(figure, out_dir / "02_numeric_distributions.png",
            f"Numeric feature distributions — top {len(columns)} by |correlation with '{label}'|{note}")


def chart_categorical_counts(df, label, out_dir):
    columns = categorical_columns(df)[:GRID_CAP]
    skipped = len(categorical_columns(df)) - len(columns)
    if not columns:
        return
    rows = math_rows(len(columns))
    figure, axes = _grid(rows, 4)
    for axis in axes:
        axis.set_visible(False)
    for axis, column in zip(axes, columns):
        axis.set_visible(True)
        counts = df[column].value_counts(dropna=False).head(8)
        axis.bar([str(value)[:14] for value in counts.index], counts.values, color=BLUE,
                 edgecolor="white", linewidth=0.4)
        axis.set_title(column, fontsize=10)
        axis.tick_params(axis="x", labelsize=7, rotation=30)
    note = f" · {skipped} more not shown" if skipped > 0 else ""
    _finish(figure, out_dir / "03_categorical_counts.png",
            f"Categorical counts (cardinality ≤ {MAX_CATEGORICAL_CARDINALITY}, top 8 values each){note}")


def chart_label_correlation(df, label, out_dir):
    correlations = label_correlations(df, df[label])
    if correlations is None:
        return
    top = correlations.head(TOP_N)[::-1]
    figure, axis = plt.subplots(figsize=(8, 0.42 * len(top) + 2.0))
    axis.barh([name[:30] for name in top.index], top.values,
              color=[RED if value < 0 else GREEN for value in top.values])
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Pearson correlation with the label (positive = raises the label)")
    axis.set_title(f"Which features move '{label}'", fontsize=12)
    axis.margins(y=0.02)
    figure.tight_layout()
    figure.savefig(out_dir / "04_label_correlation.png", dpi=110)
    plt.close(figure)


def chart_correlation_heatmap(df, label, out_dir):
    correlations = label_correlations(df, df[label])
    if correlations is None:
        return
    columns = list(correlations.index[:TOP_N])
    encode, encoded = label_encoding(df[label])
    numeric = set(numeric_columns(df))
    columns = [column for column in columns if column in numeric]
    frame = df[columns].copy()
    frame[label] = encoded.reindex(frame.index)
    matrix = frame.corr(numeric_only=True)
    figure, axis = plt.subplots(figsize=(0.55 * len(matrix) + 3.5, 0.55 * len(matrix) + 2.5))
    image = axis.imshow(matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(matrix)))
    axis.set_yticks(range(len(matrix)))
    axis.set_xticklabels(list(matrix.columns) + [label] if label not in matrix.columns else list(matrix.columns),
                         rotation=45, ha="right", fontsize=7)
    axis.set_yticklabels(list(matrix.columns), fontsize=7)
    for row in range(len(matrix)):
        for column in range(len(matrix)):
            axis.annotate(f"{matrix.iloc[row, column]:.2f}", (column, row),
                          ha="center", va="center", fontsize=5.5,
                          color="white" if abs(matrix.iloc[row, column]) > 0.5 else "black")
    figure.colorbar(image, shrink=0.8)
    axis.set_title(f"Top {len(matrix) - 1} numeric features + '{label}' — correlation matrix", fontsize=11)
    figure.tight_layout()
    figure.savefig(out_dir / "05_correlation_heatmap.png", dpi=110)
    plt.close(figure)


def chart_missing_values(df, label, out_dir):
    missing = df.isna().mean().mul(100).sort_values()
    figure, axis = plt.subplots(figsize=(10, max(0.35 * len(missing) + 2, 4)))
    axis.barh([name[:30] for name in missing.index], missing.values, color=ORANGE)
    axis.set_xlabel("% of rows missing")
    axis.set_title("Missing values per column — " +
                   ("no missing values at all" if missing.max() == 0 else f"worst: {missing.idxmax()} ({missing.max():.1f}%)"),
                   fontsize=12)
    axis.margins(y=0.02)
    figure.tight_layout()
    figure.savefig(out_dir / "06_missing_values.png", dpi=110)
    plt.close(figure)


def chart_group_selection_rate(df, label, sensitive, out_dir):
    encode, encoded = label_encoding(df[label])
    if encoded is None or sensitive not in df.columns:
        # bank_marketing declares age_group but only ships a raw `age` column
        if sensitive == "age_group" and "age" in df.columns:
            df = data.add_age_group(df)
        else:
            return
    if sensitive not in df.columns:
        return
    frame = pd.DataFrame({"group": df[sensitive], "positive": encoded})
    frame = frame.dropna(subset=["group"])
    rates = frame.groupby("group")["positive"].agg(["mean", "count"])
    figure, axis = plt.subplots(figsize=(8, 0.55 * len(rates) + 2.5))
    colors = [GREY if index == "all" else BLUE for index in rates.index]
    bars = axis.barh([str(value)[:22] for value in rates.index], rates["mean"], color=colors)
    axis.axvline(encoded.mean(), color=RED, linestyle="--", linewidth=1.2,
                 label=f"overall {encoded.mean():.0%}")
    for bar, (rate, count) in zip(bars, zip(rates["mean"], rates["count"])):
        text = f"{rate:.0%}  (n={count:,})"
        if rate >= 0.75:  # high bars: label inside, so it never spills past the axes
            axis.annotate(text, (rate - 0.015, bar.get_y() + bar.get_height() / 2),
                          va="center", ha="right", fontsize=9, color="white", fontweight="bold")
        else:
            axis.annotate(text, (rate, bar.get_y() + bar.get_height() / 2),
                          va="center", ha="left", fontsize=9, xytext=(4, 0), textcoords="offset points")
    axis.set_xlim(0, 1.0)
    axis.set_xlabel(f"P({label} = 1)")
    axis.set_title(f"Positive-label rate by '{sensitive}' — the fairness lens", fontsize=12)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=9, ncols=3, frameon=False)
    axis.invert_yaxis()
    figure.tight_layout()
    figure.savefig(out_dir / "07_group_selection_rate.png", dpi=110)
    plt.close(figure)


def chart_boxplots(df, label, out_dir):
    columns = [column for column in top_numeric_by_label(df, df[label], 7) if column != label][:6]
    if not columns:
        return
    encode, encoded = label_encoding(df[label])
    figure, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = np.atleast_1d(axes).ravel()
    for axis in axes:
        axis.set_visible(False)
    classes = sorted(str(value) for value in encoded.dropna().unique()) if encode else None
    for axis, column in zip(axes, columns):
        axis.set_visible(True)
        series = df[column]
        groups = []
        if encode is not None:
            labels_present = encoded.notna()
            negatives = series[labels_present & (encoded == 0)].dropna()
            positives = series[labels_present & (encoded == 1)].dropna()
            if len(negatives) and len(positives):
                groups = [negatives, positives]
                texts = ["negatives", "positives"]
            else:
                groups = [series.dropna()]
                texts = ["all rows"]
        else:
            groups = [series.dropna()]
            texts = ["all rows"] if encode is None else []
        if groups:
            axis.boxplot(groups, tick_labels=texts, vert=False, showfliers=False)
        axis.set_title(column, fontsize=10)
        axis.tick_params(labelsize=8)
    names = f"{classes[0]} / {classes[1]}" if classes else ""
    _finish(figure, out_dir / "08_boxplots_top_features.png",
            f"Box plots of the top {len(columns)} numeric features — negative vs positive class {names} (outliers hidden)")


def chart_scatter(df, label, out_dir):
    columns = [column for column in top_numeric_by_label(df, df[label], 4) if column != label][:2]
    if len(columns) < 2:
        return
    encode, encoded = label_encoding(df[label])
    if encode is None:
        return
    sample = df[[columns[0], columns[1]]].join(encoded.rename("_label")).dropna()
    if len(sample) > 1000:
        sample = sample.sample(1000, random_state=42)
    neg_name = [value for value, code in encode.items() if code == 0][0]
    pos_name = [value for value, code in encode.items() if code == 1][0]
    figure, axis = plt.subplots(figsize=(8, 6))
    for positive, color, name in ((0, GREY, neg_name), (1, ORANGE, pos_name)):
        subset = sample[sample["_label"] == positive]
        axis.scatter(subset[columns[0]], subset[columns[1]], s=7, alpha=0.35, color=color,
                     label=f"{label} = {name}")
    axis.set_xlabel(columns[0])
    axis.set_ylabel(columns[1])
    axis.set_title(f"{columns[0]} vs {columns[1]} — 1,000-row sample coloured by '{label}'", fontsize=12)
    axis.legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(out_dir / "09_scatter_top_features.png", dpi=110)
    plt.close(figure)


def math_rows(count: int) -> int:
    return int(np.ceil(count / 4))


CHARTS = [
    ("01_label_distribution.png", chart_label_distribution),
    ("02_numeric_distributions.png", chart_numeric_distributions),
    ("03_categorical_counts.png", chart_categorical_counts),
    ("04_label_correlation.png", chart_label_correlation),
    ("05_correlation_heatmap.png", chart_correlation_heatmap),
    ("06_missing_values.png", chart_missing_values),
    ("08_boxplots_top_features.png", chart_boxplots),
    ("09_scatter_top_features.png", chart_scatter),
]


def visualize(dataset, label, sensitive) -> dict:
    path = config.DATA_DIR / f"{dataset}.csv"
    if not path.exists():
        return {"dataset": dataset, "charts": 0, "note": "CSV not found — run scripts/download_datasets.py"}
    df = data.load_csv(path)
    if label not in df.columns:
        return {"dataset": dataset, "charts": 0, "note": f"label '{label}' not in columns"}
    out_dir = OUT_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    # the fairness lens needs the sensitive column, so it is not part of CHARTS
    try:
        chart_group_selection_rate(df, label, sensitive, out_dir)
        if (out_dir / "07_group_selection_rate.png").exists():
            written.append("07_group_selection_rate.png")
    except Exception as error:
        print(f"  ! {dataset}/07_group_selection_rate.png: {type(error).__name__}: {error}")
    for name, function in CHARTS:
        try:
            function(df, label, out_dir)
            if (out_dir / name).exists():
                written.append(name)
        except Exception as error:  # one bad column must not kill the dataset suite
            print(f"  ! {dataset}/{name}: {type(error).__name__}: {error}")
    print(f"{dataset:20s} {len(written):2d} charts -> {out_dir}")
    return {"dataset": dataset, "charts": len(written), "note": ""}


def visualize_drift():
    base = config.DATA_DIR / "demo_loans.csv"
    new = config.DATA_DIR / "demo_loans_new.csv"
    if not base.exists() or not new.exists():
        return None
    reference = data.load_csv(base)
    current = data.load_csv(new)
    result = drift.assess_drift(reference, current)
    out_dir = OUT_DIR / "drift"
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(result.items(), key=lambda item: -item[1]["psi"])
    names = [f"{name[:24]} ({entry['psi_level'][0]})" for name, entry in ordered]
    values = [entry["psi"] for _, entry in ordered]
    colors = [LEVEL_COLOR.get(entry["psi_level"], GREY) for _, entry in ordered]
    figure, axis = plt.subplots(figsize=(10, max(0.4 * len(names) + 2, 4)))
    axis.barh(names, values, color=colors)
    for level, color in LEVEL_COLOR.items():
        axis.scatter([], [], color=color, label=level)
    axis.axvline(0.1, color=ORANGE, linestyle="--", linewidth=1)
    axis.axvline(0.25, color=RED, linestyle="--", linewidth=1)
    axis.set_xlabel("Population Stability Index (dashed lines: 0.10 moderate / 0.25 major)")
    axis.set_title("demo_loans vs demo_loans_new — PSI per feature", fontsize=12)
    axis.legend(fontsize=9)
    axis.margins(y=0.02)
    axis.invert_yaxis()
    figure.tight_layout()
    figure.savefig(out_dir / "10_drift_demo_loans.png", dpi=110)
    plt.close(figure)
    print(f"{'drift (demo pair)':20s}  1 chart  -> {out_dir}")
    return out_dir / "10_drift_demo_loans.png"


def visualize_text():
    path = config.DATA_DIR / "sample_texts.csv"
    if not path.exists():
        return None
    texts = data.load_csv(path)["text"].dropna().tolist()
    result = llm_eval.evaluate_texts(texts)
    out_dir = OUT_DIR / "text"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = result.get("kind_summary") or {}
    figure, axis = plt.subplots(figsize=(8, 5))
    names = list(summary)
    values = [summary[name] for name in names]
    axis.bar(names, values, color=[RED if value >= result.get("flag_threshold", 0.5) else GREEN for value in values])
    axis.axhline(result.get("flag_threshold", 0.5), color=RED, linestyle="--", linewidth=1, label="flag threshold")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("mean judge score (0 safe … 1 risky)")
    axis.set_title(f"Heuristic text judges on sample_texts.csv · backend: {result.get('backend')}", fontsize=12)
    axis.legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(out_dir / "11_text_safety.png", dpi=110)
    plt.close(figure)
    print(f"{'text (sample_texts)':20s}  1 chart  -> {out_dir}")
    return out_dir / "11_text_safety.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", default=None,
                        help="dataset stem to visualize (default: every dataset in DATASETS)")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for dataset, (label, sensitive) in DATASETS.items():
        if args.only and dataset != args.only:
            continue
        results.append(visualize(dataset, label, sensitive))
    if not args.only:
        visualize_drift()
        visualize_text()
    print(f"\nDone — browse visualize/README.md")


if __name__ == "__main__":
    main()
