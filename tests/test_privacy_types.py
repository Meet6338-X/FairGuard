"""Tests for the type-aware privacy scan — RC-08.

A 16-digit ``account_id`` used to match the credit-card pattern *and* the
phone pattern (severity 1.0), and a plain numeric id matched the phone
pattern, so a dataset of harmless identifiers was reported as carrying
credit-card numbers.
"""
import pandas as pd

from fairguard import privacy


def test_plain_numeric_id_is_not_a_credit_card():
    df = pd.DataFrame({"account_id": [1234567890123456, 9876543210987654]})
    result = privacy.scan_pii(df)
    types = {f["type"] for f in result["findings"]}
    assert "credit_card" not in types
    assert "phone" not in types


def test_numeric_id_column_is_still_flagged_by_its_name():
    """The digits are harmless, but the column *name* is a hint."""
    df = pd.DataFrame({"account_id": [1234567890123456, 9876543210987654]})
    result = privacy.scan_pii(df)
    types = {f["type"] for f in result["findings"]}
    assert "sensitive_column_name" in types


def test_numeric_column_named_card_number_is_scanned_as_a_card():
    df = pd.DataFrame({"card_number": [4111111111111111, 5500000000000004]})
    result = privacy.scan_pii(df)
    types = {f["type"] for f in result["findings"]}
    assert "credit_card" in types


def test_numeric_column_named_phone_is_scanned_as_a_phone():
    df = pd.DataFrame({"phone_number": [9876543210, 9123456789]})
    result = privacy.scan_pii(df)
    types = {f["type"] for f in result["findings"]}
    assert "phone" in types


def test_text_columns_are_always_scanned():
    """Free text is where a card number really hides."""
    df = pd.DataFrame({"notes": ["pay with 4111111111111111 please"]})
    result = privacy.scan_pii(df)
    types = {f["type"] for f in result["findings"]}
    assert "credit_card" in types


def test_ip_address_column_is_detected():
    df = pd.DataFrame({"ip": ["192.168.1.1", "10.0.0.5"]})
    result = privacy.scan_pii(df)
    types = {f["type"] for f in result["findings"]}
    assert "ip_address" in types


def test_email_column_is_detected():
    df = pd.DataFrame({"email": ["a.b@example.com", "c.d@example.org"]})
    result = privacy.scan_pii(df)
    types = {f["type"] for f in result["findings"]}
    assert "email" in types


def test_harmless_numeric_columns_produce_no_findings():
    df = pd.DataFrame({"income": [40000, 52000], "age": [31, 45], "score": [0.7, 0.2]})
    result = privacy.scan_pii(df)
    assert result["findings"] == []
    assert result["risk"] == 0.0


def test_risk_does_not_saturate_on_many_weak_findings():
    """Noisy-OR combination, so weak findings do not add up to 1.0."""
    findings = [
        {"matches": 1, "severity": 0.2},
        {"matches": 1, "severity": 0.2},
        {"matches": 1, "severity": 0.2},
    ]
    risk = privacy.privacy_risk(findings, rows=1000)
    assert 0 < risk < 0.2


def test_samples_are_masked():
    df = pd.DataFrame({"email": ["secret.person@example.com"]})
    result = privacy.scan_pii(df)
    sample = result["findings"][0]["samples"][0]
    assert "secret" not in sample
    assert "*" in sample
