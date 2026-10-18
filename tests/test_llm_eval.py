"""Tests for fairguard.llm_eval — text evaluation summary."""
from fairguard import llm_eval


def test_evaluate_texts_summary_shape():
    result = llm_eval.evaluate_texts(["you are stupid", "have a nice day"])
    assert result["n_texts"] == 2
    assert set(result["kind_summary"]) == {"toxicity", "bias", "hate_speech"}
    assert len(result["per_text"]) == 2
    assert 0 <= result["risk"] <= 1
    assert isinstance(result["flagged"], int)


def test_evaluate_texts_flags_bad_text():
    result = llm_eval.evaluate_texts(["you are stupid"])
    assert result["flagged"] == 1
    assert result["per_text"][0]["flagged"] is True
    assert result["risk"] == 1.0


def test_evaluate_texts_with_references_adds_hallucination():
    result = llm_eval.evaluate_texts(
        ["the cat sat on the mat"],
        references=["the cat sat on the mat"],
    )
    assert "hallucination" in result["kind_summary"]
    assert result["kind_summary"]["hallucination"] == 0.0


def test_text_risk_accessor():
    assert llm_eval.text_risk({"risk": 0.42}) == 0.42
    assert llm_eval.text_risk({}) == 0.0
