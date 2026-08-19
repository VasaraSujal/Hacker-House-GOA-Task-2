from __future__ import annotations

import re
import time

from rag.generation.base import GenerationResult, LLMProvider
from rag.generation.prompts import REFUSAL_MESSAGE
from rag.retrieval.bm25 import extract_content_tokens, tokenize


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।])\s+|\n+")

_NON_LATIN_NON_DEVA = re.compile(
    r"[\u0A80-\u0AFF\u0980-\u09FF\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u0A00-\u0A7F\u0B00-\u0B7F\u0600-\u06FF]"
)


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p and p.strip()]
    if parts:
        return parts
    cleaned = text.strip()
    return [cleaned] if cleaned else []


def _overlap_score(query_tokens: set[str], sentence: str) -> float:
    sent_tokens = set(tokenize(sentence))
    if not query_tokens or not sent_tokens:
        return 0.0
    inter = len(query_tokens & sent_tokens)
    return inter / float(len(query_tokens))


class ExtractiveAnswerProvider(LLMProvider):
    """Compose an answer only from retrieved context sentences (no generative model)."""

    def __init__(self, *, max_sentences: int = 3, min_overlap: float = 0.05) -> None:
        self.max_sentences = max_sentences
        self.min_overlap = min_overlap

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> GenerationResult:
        del system_prompt
        started = time.perf_counter()
        query, context = self._parse_prompt(prompt)
        query_tokens = extract_content_tokens(query)
        scored: list[tuple[float, str]] = []
        blocks = self._context_blocks(context)
        for block in blocks:
            for sentence in _split_sentences(block):
                score = _overlap_score(query_tokens, sentence)
                if score >= self.min_overlap:
                    scored.append((score, sentence))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected: list[str] = []
        seen: set[str] = set()
        for _, sentence in scored:
            key = sentence.casefold()
            if key in seen:
                continue
            seen.add(key)
            selected.append(sentence)
            if len(selected) >= self.max_sentences:
                break

        # Cross-script fallback: if no sentence had lexical token overlap (e.g. query in Gujarati,
        # context in English/Hindi), but context was retrieved and verified by the guardrails,
        # extract the top sentences from the highest-ranked context block.
        if not selected and blocks and _NON_LATIN_NON_DEVA.search(query):
            for sentence in _split_sentences(blocks[0]):
                key = sentence.casefold()
                if key not in seen:
                    seen.add(key)
                    selected.append(sentence)
                if len(selected) >= self.max_sentences:
                    break

        if not selected:
            text = REFUSAL_MESSAGE
        else:
            text = " ".join(selected)
        return GenerationResult(
            text=text,
            latency_ms=(time.perf_counter() - started) * 1000,
            model="extractive",
            raw={"n_sentences": len(selected)},
        )

    @staticmethod
    def _parse_prompt(prompt: str) -> tuple[str, str]:
        question = ""
        context = ""
        if "Question:\n" in prompt and "\n\nRetrieved context:\n" in prompt:
            after_q = prompt.split("Question:\n", 1)[1]
            question, rest = after_q.split("\n\nRetrieved context:\n", 1)
            context = rest.split("\n\nAnswer the question", 1)[0]
        else:
            context = prompt
        return question.strip(), context.strip()

    @staticmethod
    def _context_blocks(context: str) -> list[str]:
        if not context:
            return []
        blocks: list[str] = []
        # Prefer the payload text sections emitted by ContextBuilder.
        for match in re.finditer(r"Text:\n(.*?)(?=\n\n\[Source|\Z)", context, flags=re.S):
            text = match.group(1).strip()
            if text:
                blocks.append(text)
        if blocks:
            return blocks
        raw = re.split(r"\n\[(?:source|chunk|doc)[^\]]*\]\n|\n---\n|\n\n+", context, flags=re.I)
        return [b.strip() for b in raw if b and b.strip()]
