import pytest

from app.core.exceptions import InvalidQueryError
from rag.generation.prompts import REFUSAL_MESSAGE
from rag.guardrails.grounding_guard import GroundingGuard
from rag.guardrails.input_guard import InputGuard
from rag.guardrails.relevance_guard import RelevanceGuard
from rag.retrieval.types import RetrievalResult


def test_input_guard_rejects_empty() -> None:
    guard = InputGuard(max_chars=20)
    with pytest.raises(InvalidQueryError):
        guard.validate("   ")


def test_input_guard_rejects_too_long() -> None:
    guard = InputGuard(max_chars=8)
    with pytest.raises(InvalidQueryError):
        guard.validate("this is way too long")


def test_input_guard_accepts_normal() -> None:
    assert InputGuard().validate("What is Paris?") == "What is Paris?"


def test_relevance_guard_empty() -> None:
    decision = RelevanceGuard(min_score=0.2).check([])
    assert not decision.ok


def test_relevance_guard_low_score() -> None:
    hits = [RetrievalResult("x", 0.01, "d", "c")]
    decision = RelevanceGuard(min_score=0.2).check(hits)
    assert not decision.ok
    assert decision.reason == "low_score"


def test_relevance_guard_ok() -> None:
    hits = [RetrievalResult("x", 0.9, "d", "c")]
    assert RelevanceGuard(min_score=0.2).check(hits).ok


def test_grounding_accepts_refusal() -> None:
    decision = GroundingGuard().check(REFUSAL_MESSAGE, "irrelevant")
    assert decision.grounded


def test_grounding_rejects_ungrounded() -> None:
    decision = GroundingGuard(min_overlap=0.3).check(
        "The secret launch code is zebra-nine.",
        "Paris is the capital of France.",
    )
    assert not decision.grounded


def test_grounding_accepts_overlap() -> None:
    decision = GroundingGuard(min_overlap=0.2).check(
        "Paris is the capital of France.",
        "Paris is the capital and largest city of France.",
    )
    assert decision.grounded
