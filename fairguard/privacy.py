"""Privacy assessment — find PII / sensitive columns, then summarise the risk."""
import re

import pandas as pd

from . import config

# type → (pattern, severity 0..1). Severities are editorial weights.
PATTERNS = {
    "email": (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), 0.9),
    "phone": (re.compile(r"(?:\+91[\s-]?)?[6-9]\d{9}\b"), 0.8),
    "credit_card": (re.compile(r"\b(?:\d[ -]?){13,16}\b"), 1.0),
    "aadhaar": (re.compile(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b"), 1.0),
    "ip_address": (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), 0.4),
}
NAME_HINTS = ("name", "email", "phone", "mobile", "address", "aadhaar", "ssn",
              "passport", "pan", "dob", "birth", "account", "ifsc", "upi")
NAME_HINT_SEVERITY = 0.7


def mask(value) -> str:
    """Keep the first two characters, hide the rest — safe for reports."""
    text = str(value)
    return text[:2] + "*" * max(0, len(text) - 2)


def scan_pii(df: pd.DataFrame) -> dict:
    """Detect PII-ish columns and return findings plus a 0..1 privacy risk."""
    findings = []
    for column in df.columns:
        values = df[column].dropna().astype(str)
        if values.empty:
            continue
        regex_hit = False
        for pii_type, (pattern, severity) in PATTERNS.items():
            hits = values[values.str.contains(pattern, regex=True)]
            if hits.empty:
                continue
            regex_hit = True
            findings.append({
                "column": column,
                "type": pii_type,
                "matches": int(len(hits)),
                "severity": severity,
                "samples": [mask(v) for v in hits.head(config.PII_SAMPLE_ROWS)],
            })
        if not regex_hit and any(hint in column.lower() for hint in NAME_HINTS):
            findings.append({
                "column": column,
                "type": "sensitive_column_name",
                "matches": int(len(values)),
                "severity": NAME_HINT_SEVERITY,
                "samples": [mask(v) for v in values.head(config.PII_SAMPLE_ROWS)],
            })
    return {"findings": findings, "risk": privacy_risk(findings, len(df))}


def privacy_risk(findings: list, rows: int) -> float:
    """Severity-weighted share of affected rows, capped at 1."""
    if not findings or rows == 0:
        return 0.0
    score = sum(f["severity"] * min(f["matches"] / rows, 1.0) for f in findings)
    return float(min(score, 1.0))
