"""Tests for fairguard.privacy."""
import pandas as pd

from fairguard import privacy


def test_scan_pii_finds_email_and_name_columns(demo_df):
    result = privacy.scan_pii(demo_df)
    assert result["findings"], "demo data contains PII on purpose"
    by_column = {f["column"]: f for f in result["findings"]}
    assert by_column["email"]["type"] == "email"
    assert by_column["full_name"]["type"] == "sensitive_column_name"
    assert 0 < result["risk"] <= 1


def test_findings_are_masked(demo_df):
    result = privacy.scan_pii(demo_df)
    for finding in result["findings"]:
        assert all("*" in sample for sample in finding["samples"])


def test_aadhaar_pattern_detected():
    df = pd.DataFrame({"aadhaar_no": ["2345 6789 0123", "2345 6789 0124"]})
    result = privacy.scan_pii(df)
    assert result["findings"][0]["type"] == "aadhaar"
    assert result["findings"][0]["severity"] == 1.0


def test_mask_and_empty_cases():
    assert privacy.mask("hello") == "he***"
    assert privacy.mask("a") == "a"
    assert privacy.privacy_risk([], 100) == 0.0
    assert privacy.privacy_risk([{"severity": 0.9, "matches": 50}], 100) == 0.45
