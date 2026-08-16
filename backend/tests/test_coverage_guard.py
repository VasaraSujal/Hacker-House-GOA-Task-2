from __future__ import annotations

from rag.guardrails.coverage_guard import LexicalCoverageGuard
from rag.retrieval.types import RetrievalResult


def test_lexical_coverage_refuses_cricket_against_unrelated_docs() -> None:
    guard = LexicalCoverageGuard(min_overlap=0.34)
    results = [
        RetrievalResult(
            "Audrey Hepburn died yesterday at her home in Switzerland.",
            0.8,
            "d1",
            "c1",
        )
    ]
    out = guard.check("Who won yesterday's cricket match?", results)
    assert out.ok is False


def test_lexical_coverage_accepts_corporation_query() -> None:
    guard = LexicalCoverageGuard(min_overlap=0.34)
    results = [
        RetrievalResult(
            "A corporation is a company authorized to act as a single legal entity.",
            0.8,
            "d1",
            "c1",
        )
    ]
    out = guard.check("What is a corporation?", results)
    assert out.ok is True
