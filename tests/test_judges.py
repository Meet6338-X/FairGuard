"""Tests for fairguard.judges — the offline heuristic backend."""
from fairguard import judges


def test_toxicity_scores_bad_text():
    judge = judges.HeuristicJudge()
    assert judge.score("This is a stupid useless service", "toxicity") > 0.5
    assert judge.score("The weather is lovely today", "toxicity") == 0.0


def test_hate_speech_words():
    judge = judges.HeuristicJudge()
    assert judge.score("I hate those people", "hate_speech") == 1.0


def test_bias_phrases():
    judge = judges.HeuristicJudge()
    assert judge.score("All women should stay home", "bias") == 1.0
    assert judge.score("This applicant has a stable income", "bias") == 0.0


def test_hallucination_needs_reference():
    judge = judges.HeuristicJudge()
    reference = "the cat sat quietly on the warm mat"
    assert judge.score(reference, "hallucination", reference=reference) == 0.0
    assert judge.score("a dog flew over the moon", "hallucination", reference=reference) == 1.0
    assert judge.score("anything at all", "hallucination") == 0.0  # no reference -> no verdict


def test_pii_detection_in_text():
    judge = judges.HeuristicJudge()
    assert judge.score("call me at 9876543210 now", "pii") == 0.8
    assert judge.score("nothing personal here", "pii") == 0.0


def test_get_judge_backends():
    judge = judges.get_judge("toxicity", backend="heuristic")
    assert isinstance(judge, judges.HeuristicJudge)
    auto = judges.get_judge("toxicity")  # heuristic when transformers is absent
    if not judges.transformers_available():
        assert auto.name == "heuristic"
