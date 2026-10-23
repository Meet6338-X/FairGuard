"""Dataset loading, validation and profiling."""
import pandas as pd


def load_csv(path) -> pd.DataFrame:
    """Load a CSV dataset from disk."""
    return pd.read_csv(path)


def add_age_group(df: pd.DataFrame, column: str = "age") -> pd.DataFrame:
    """Return a copy with a binned age column — handy as a sensitive attribute.

    If the dataset has no `age` column, returns df unchanged.  Ages outside
    the bins become the string ``"unknown"`` instead of NaN, so the column
    never silently loses rows in a fairness group.
    """
    out = df.copy()
    if column not in out.columns:
        return out
    binned = pd.cut(out[column], bins=(0, 30, 45, 200), labels=("<30", "30-45", ">45"))
    out["age_group"] = binned.astype(object).where(binned.notna(), "unknown").astype(str)
    return out


def is_binary_label(values) -> bool:
    """True when the label column has at most two distinct values.

    RC-09: the old check was ``values <= {0, 1}``, which (a) rejected valid
    string labels such as "yes"/"no" and (b) raised a ``TypeError`` on
    mixed-type labels.  A label with a single distinct value is still
    accepted — it is degenerate, but that is a modelling problem, not a
    validation problem.
    """
    series = pd.Series(values).dropna()
    if series.empty:
        return False
    return len(pd.unique(series)) <= 2


def binary_label_mapping(values) -> dict:
    """Map a binary label column onto ``{original_value: 0 or 1}``.

    Accepts ``0/1``, ``1/0``, ``True/False`` and two arbitrary strings
    (``yes/no``, ``approved/denied``, ...).  The *positive* class is the one
    that sorts second, so the mapping is deterministic.  Returns ``{}`` when
    the column does not have exactly two distinct values.
    """
    series = pd.Series(values).dropna()
    uniques = list(pd.unique(series))
    if len(uniques) != 2:
        return {}
    try:
        ordered = sorted(uniques)
    except TypeError:  # mixed types — fall back to appearance order
        ordered = uniques
    negative, positive = ordered[0], ordered[1]
    mapping = {negative: 0, positive: 1}
    # booleans: True must mean 1
    if negative is True or positive is False:
        mapping = {False: 0, True: 1}
    return mapping


def encode_label(df: pd.DataFrame, label: str) -> tuple:
    """Return ``(df, mapping)`` with the label column encoded as 0/1.

    Already-numeric 0/1 labels are left untouched (mapping ``{0: 0, 1: 1}``).
    """
    if label not in df.columns:
        return df, {}
    mapping = binary_label_mapping(df[label])
    if not mapping:
        return df, {}
    out = df.copy()
    out[label] = out[label].map(mapping)
    out[label] = out[label].fillna(0).astype(int)
    return out, mapping


def validate_dataset(df, sensitive, label="approved", model_features=None) -> list:
    """Return human-readable problems with the dataset; empty list means OK."""
    problems = []
    if df.empty:
        return ["dataset is empty"]
    if label not in df.columns:
        problems.append(f"label column '{label}' not found")
    for col in sensitive:
        if col not in df.columns:
            problems.append(f"sensitive column '{col}' not found")
    if len(df) < 50:
        problems.append(f"dataset is very small ({len(df)} rows)")
    for col in df.columns[df.isna().mean() > 0.5]:
        problems.append(f"column '{col}' is more than 50% missing")
    if label in df.columns:
        # RC-09: the old check (`values <= {0, 1}` then `sorted(values)`) raised
        # a TypeError on mixed-type labels and rejected perfectly valid string
        # labels such as "yes"/"no".
        if not is_binary_label(df[label]):
            uniques = list(pd.unique(df[label].dropna()))
            shown = ", ".join(repr(u) for u in uniques[:5])
            problems.append(f"label '{label}' is not binary - found {shown}")
    if model_features is not None and len(model_features) > 0:
        # Encoded features like 'gender_female' are fine when 'gender' exists.
        absent = [
            c for c in model_features
            if c not in df.columns and c.rsplit("_", 1)[0] not in df.columns
        ]
        if absent:
            problems.append(f"model feature columns missing: {absent}")
    return problems


def profile_dataset(df: pd.DataFrame) -> dict:
    """Small dataset profile: shape, missing %, numeric ranges, value counts."""
    profile = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_pct": (df.isna().mean() * 100).round(2).to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_summary": {},
        "categorical_summary": {},
    }
    for col in df.select_dtypes("number").columns:
        profile["numeric_summary"][col] = {
            "min": round(float(df[col].min()), 2),
            "max": round(float(df[col].max()), 2),
            "mean": round(float(df[col].mean()), 2),
            "std": round(float(df[col].std()), 2),
        }
    for col in df.columns:
        unusual = not pd.api.types.is_numeric_dtype(df[col]) or df[col].nunique() <= 10
        if unusual:
            counts = df[col].value_counts(dropna=False).head(10)
            profile["categorical_summary"][col] = {str(k): int(v) for k, v in counts.items()}
    return profile
