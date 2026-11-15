"""Per-example descriptors saved next to every prediction.

They make the revision analyses reproducible from prediction files alone:
claim-count vs false-positive rate, answer-length shortcuts, source/target
breakdowns, and cluster (question/context group) bootstraps.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence

from clinhallu.data.leakage import canonical_group_key

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
DEFAULT_SOURCE_PREFIXES = ("c1r",)


def answer_claims(answer: str) -> list[str]:
    """Sentence-level claim proxy: sentences with at least three words.

    This is a deliberately simple, deterministic proxy (no model involved).  Its
    validity for the claim-count hypothesis should be spot-checked by hand.
    """
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(str(answer).strip()) if part.strip()]
    claims = [part for part in parts if len(part.split()) >= 3]
    return claims or ([str(answer).strip()] if str(answer).strip() else [])


def source_tag(sample_id, prefixes: Sequence[str] = DEFAULT_SOURCE_PREFIXES) -> str:
    text = str(sample_id)
    return "source" if any(text.startswith(prefix) for prefix in prefixes) else "target"


def example_metadata(
    sample: Mapping,
    *,
    source_prefixes: Sequence[str] = DEFAULT_SOURCE_PREFIXES,
    context_mode: str = "normal",
) -> dict:
    answer = str(sample.get("answer", ""))
    sample_id = sample.get("sample_id", sample.get("id", ""))
    return {
        "group_key": canonical_group_key(sample)[:16],
        "source_tag": source_tag(sample_id, source_prefixes),
        "context_mode": context_mode,
        "question_words": len(str(sample.get("question", "")).split()),
        "context_words": len(str(sample.get("context", "")).split()),
        "answer_words": len(answer.split()),
        "answer_sentences": len([p for p in _SENTENCE_SPLIT.split(answer.strip()) if p.strip()]),
        "claim_count": len(answer_claims(answer)),
    }


def source_prefixes_from_config(cfg: Mapping) -> tuple[str, ...]:
    value = cfg.get("data", {}).get("source_id_prefixes")
    return tuple(value) if value else DEFAULT_SOURCE_PREFIXES
