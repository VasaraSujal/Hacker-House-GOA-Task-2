from __future__ import annotations

import re
from dataclasses import dataclass

from rag.generation.prompts import REFUSAL_MESSAGE

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 2}


@dataclass(slots=True)
class GroundingDecision:
    grounded: bool
    overlap: float
    reason: str


class GroundingGuard:
    """Lightweight lexical overlap check between the answer and retrieved context.

    Extensible later with a model-based NLI/attribution validator.
    """

    def __init__(self, min_overlap: float = 0.12, refusal_message: str = REFUSAL_MESSAGE) -> None:
        self.min_overlap = min_overlap
        self.refusal_message = refusal_message

    def check(self, answer: str, context: str) -> GroundingDecision:
        normalized = " ".join((answer or "").split())
        if not normalized:
            return GroundingDecision(False, 0.0, "empty_answer")
        if normalized == self.refusal_message or normalized.lower() == self.refusal_message.lower():
            return GroundingDecision(True, 1.0, "explicit_refusal")
        answer_tokens = _tokens(normalized)
        context_tokens = _tokens(context)
        if not answer_tokens:
            return GroundingDecision(False, 0.0, "no_answer_tokens")
        if not context_tokens:
            return GroundingDecision(False, 0.0, "no_context")
        overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
        if overlap >= self.min_overlap:
            return GroundingDecision(True, overlap, "supported")
        return GroundingDecision(False, overlap, "unsupported")
