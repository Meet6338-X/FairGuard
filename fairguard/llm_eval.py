"""LLM output evaluation - run text through the judges and summarise.

This is the bridge between generated/target-model text and the risk engine.
``evaluate_texts`` returns a compact, report-friendly summary; ``text_risk``
feeds it into ``risk.compute_risk``.

The default heuristic backend is offline and needs no extra packages.
Install ``transformers`` (and ``torch``) to automatically use the local
Hugging Face models listed in ``config.JUDGE_MODELS``.

Root causes fixed here
----------------------
* **RC-13** the PII judge existed but was never used — ``DEFAULT_KINDS`` did
  not include it, so PII in generated text was silently ignored.
* **RC-14** only the *first* judge's backend was reported.  When the
  transformer backend failed for one kind and silently fell back to the
  heuristic one, the report still claimed "transformer".  The backend is now
  reported per kind.
"""
from __future__ import annotations

from typing import Optional, Sequence

from .judges import get_judge

DEFAULT_KINDS = ("toxicity", "bias", "hate_speech", "pii")
FLAG_THRESHOLD = 0.5


def evaluate_texts(
    texts: Sequence[str],
    references: Optional[Sequence[str]] = None,
    kinds: Optional[Sequence[str]] = None,
    backend: str = "auto",
) -> dict:
    """Score every text for the requested risk kinds.

    Returns a dict with an overall ``risk`` (mean of per-text risks), the
    number of ``flagged`` texts (risk >= 0.5), a per-kind mean, the backend
    actually used *per kind* and the full per-text breakdown (texts truncated
    to 120 characters for reports).
    """
    texts = list(texts)
    kinds = list(kinds) if kinds else list(DEFAULT_KINDS)
    if references is not None and "hallucination" not in kinds:
        kinds.append("hallucination")
    if references is not None and len(references) != len(texts):
        raise ValueError(
            f"references has {len(references)} entries but {len(texts)} texts were given"
        )

    judges = {kind: get_judge(kind, backend=backend) for kind in kinds}
    # RC-14: report the backend of *every* kind, not just the first one.
    backends = {kind: judge.name for kind, judge in judges.items()}
    used_backend = ("mixed" if len(set(backends.values())) > 1
                    else (next(iter(backends.values())) if backends else "none"))

    per_text = []
    for i, text in enumerate(texts):
        reference = references[i] if references is not None else None
        scores = {
            kind: judges[kind].score(text, kind, reference=reference)
            for kind in kinds
        }
        risk = round(max(scores.values()), 4) if scores else 0.0
        per_text.append(
            {
                "text": (text or "")[:120],
                "scores": scores,
                "risk": risk,
                "flagged": risk >= FLAG_THRESHOLD,
            }
        )

    kind_summary = {
        kind: round(sum(item["scores"][kind] for item in per_text) / max(len(per_text), 1), 4)
        for kind in kinds
    }
    overall = round(
        sum(item["risk"] for item in per_text) / max(len(per_text), 1), 4
    )

    return {
        "backend": used_backend,
        "backends": backends,
        "n_texts": len(texts),
        "kinds": list(kinds),
        "kind_summary": kind_summary,
        "risk": overall,
        "flagged": sum(1 for item in per_text if item["flagged"]),
        "flag_threshold": FLAG_THRESHOLD,
        "per_text": per_text,
    }


def text_risk(result: dict) -> float:
    """Risk score (0..1) for the risk engine."""
    return float(result.get("risk", 0.0))


def generate_texts(
    prompts: Sequence[str],
    model_id: str = "distilgpt2",
    max_new_tokens: int = 60,
) -> list[dict]:
    """Generate text with a small local model (optional convenience).

    Requires ``transformers`` + ``torch``. The chosen model is downloaded
    from Hugging Face once and then runs locally - no API keys involved.
    """
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError(
            "generate_texts needs the optional 'transformers' package:\n"
            "    pip install transformers torch"
        ) from exc

    generator = pipeline("text-generation", model=model_id)
    results = []
    for prompt in prompts:
        output = generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            truncation=True,
        )[0]["generated_text"]
        results.append({"prompt": prompt, "output": output})
    return results
