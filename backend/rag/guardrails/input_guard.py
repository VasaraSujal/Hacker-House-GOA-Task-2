from __future__ import annotations

from dataclasses import dataclass

from app.core.exceptions import InvalidQueryError


@dataclass(slots=True)
class InputGuardResult:
    ok: bool
    query: str
    reason: str | None = None


class InputGuard:
    def __init__(self, max_chars: int = 2000) -> None:
        self.max_chars = max_chars

    def validate(self, query: str | None) -> str:
        result = self.check(query)
        if not result.ok:
            raise InvalidQueryError(result.reason or "Invalid query")
        return result.query

    def check(self, query: str | None) -> InputGuardResult:
        if query is None:
            return InputGuardResult(False, "", "Query is required")
        if not isinstance(query, str):
            return InputGuardResult(False, "", "Query must be a string")
        cleaned = " ".join(query.split())
        if not cleaned:
            return InputGuardResult(False, "", "Query is empty")
        if len(cleaned) > self.max_chars:
            return InputGuardResult(False, cleaned, f"Query exceeds {self.max_chars} characters")
        if cleaned.strip("?!. ") == "":
            return InputGuardResult(False, cleaned, "Query has no meaningful content")
        return InputGuardResult(True, cleaned)
