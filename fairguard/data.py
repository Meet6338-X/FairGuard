"""Dataset loading, validation and profiling."""
import pandas as pd


def load_csv(path) -> pd.DataFrame:
    """Load a CSV dataset from disk."""
    return pd.read_csv(path)


def add_age_group(df: pd.DataFrame, column: str = "age") -> pd.DataFrame:
    """Return a copy with a binned age column — handy as a sensitive attribute.

    If the dataset has no `age` column, returns df unchanged.
    """
    out = df.copy()
    if column not in out.columns:
        return out
    out["age_group"] = pd.cut(
        out[column], bins=(0, 30, 45, 200), labels=("<30", "30-45", ">45")
    ).astype(str)
    return out


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
        values = set(df[label].dropna().unique())
        if not values <= {0, 1}:
            problems.append(f"label '{label}' is not binary — found {sorted(values)[:5]}")
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
