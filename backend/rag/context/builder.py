from __future__ import annotations

from rag.retrieval.types import RetrievalResult


class ContextBuilder:
    def __init__(self, max_chars: int = 6000) -> None:
        self.max_chars = max_chars

    def build(self, results: list[RetrievalResult]) -> tuple[str, list[RetrievalResult]]:
        selected: list[RetrievalResult] = []
        seen_text: set[str] = set()
        seen_ids: set[str] = set()
        parts: list[str] = []
        used = 0

        for result in results:
            key = " ".join(result.text.split()).lower()
            if not key or key in seen_text or result.chunk_id in seen_ids:
                continue
            block = (
                f"[Source {len(selected) + 1}]\n"
                f"Document ID: {result.document_id}\n"
                f"Chunk ID: {result.chunk_id}\n"
                f"Text:\n{result.text}"
            )
            extra = len(block) + (2 if parts else 0)
            if selected and used + extra > self.max_chars:
                break
            if extra > self.max_chars and not selected:
                truncated = result.text[: max(0, self.max_chars - 80)]
                block = (
                    f"[Source 1]\nDocument ID: {result.document_id}\n"
                    f"Chunk ID: {result.chunk_id}\nText:\n{truncated}"
                )
                parts.append(block)
                selected.append(result)
                break
            parts.append(block)
            selected.append(result)
            seen_text.add(key)
            seen_ids.add(result.chunk_id)
            used += extra

        return "\n\n".join(parts), selected
