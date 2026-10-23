"""Privacy assessment — find PII / sensitive columns, then summarise the risk.

Root cause fixed here
----------------------
**RC-08** the patterns were applied to *every* column, including numeric
ones.  A 16-digit ``account_id`` matched the credit-card pattern *and* the
phone pattern (severity 1.0), and a plain 10-digit id matched the phone
pattern — so a dataset of harmless numeric identifiers was reported as
carrying credit-card numbers.  Patterns that only make sense on free text are
now applied to numeric columns only when the column *name* justifies it
(``card_number``, ``phone``, ``aadhaar``, ...).
"""
import re

import pandas as pd

from . import config

# type → (pattern, severity 0..1). Severities are editorial weights.
PATTERNS = {
    "email": (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), 0.9),
    "phone": (re.compile(r"(?<![\d])(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"), 0.8),
    "credit_card": (re.compile(r"(?<![\d])(?:\d[ -]?){13,19}(?!\d)"), 1.0),
    "aadhaar": (re.compile(r"(?<!\d)[2-9]\d{3}\s?\d{4}\s?\d{4}(?!\d)"), 1.0),
    "ip_address": (re.compile(r"(?<![\d])(?:\d{1,3}\.){3}\d{1,3}(?!\d)"), 0.4),
}

# Patterns that only mean something on free-text columns.  On a numeric
# column the same digits are usually just an identifier (RC-08).
TEXT_ONLY_PATTERNS = {"phone", "credit_card", "aadhaar", "ip_address"}

NAME_HINTS = ("name", "email", "phone", "mobile", "address", "aadhaar", "ssn",
              "passport", "pan", "dob", "birth", "account", "ifsc", "upi")
NAME_HINT_SEVERITY = 0.7


def mask(value) -> str:
    """Keep the first two characters, hide the rest — safe for reports."""
    text = str(value)
    return text[:2] + "*" * max(0, len(text) - 2)


def _name_triggers(pii_type: str):
    return config.PII_NAME_TRIGGERS.get(pii_type, ())


def scan_pii(df: pd.DataFrame) -> dict:
    """Detect PII-ish columns and return findings plus a 0..1 privacy risk."""
    findings = []
    for column in df.columns:
        values = df[column].dropna().astype(str)
        if values.empty:
            continue
        lowered = column.lower()
        numeric_column = pd.api.types.is_numeric_dtype(df[column])
        regex_hit = False
        for pii_type, (pattern, severity) in PATTERNS.items():
            if numeric_column and pii_type in TEXT_ONLY_PATTERNS:
                # Only trust the pattern when the column name says so.
                if not any(trigger in lowered for trigger in _name_triggers(pii_type)):
                    continue
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
        if not regex_hit and any(hint in lowered for hint in NAME_HINTS):
            findings.append({
                "column": column,
                "type": "sensitive_column_name",
                "matches": int(len(values)),
                "severity": NAME_HINT_SEVERITY,
                "samples": [mask(v) for v in values.head(config.PII_SAMPLE_ROWS)],
            })
    return {
        "findings": findings,
        "risk": privacy_risk(findings, len(df)),
        "columns_scanned": int(df.shape[1]),
        "pii_columns": sorted({f["column"] for f in findings}),
    }


def privacy_risk(findings: list, rows: int) -> float:
    """Severity-weighted share of affected rows, capped at 1.

    The per-finding contributions are combined with a noisy-OR so that many
    weak findings do not saturate the scale at 1.0 the moment their sum
    passes one.
    """
    if not findings or rows == 0:
        return 0.0
    remaining = 1.0
    for finding in findings:
        share = min(finding["matches"] / rows, 1.0)
        remaining *= (1.0 - float(finding["severity"]) * share)
    return float(round(min(max(1.0 - remaining, 0.0), 1.0), 6))
