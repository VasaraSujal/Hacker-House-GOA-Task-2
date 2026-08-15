from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from app.core.exceptions import GenerationError
from rag.generation.base import GenerationResult, LLMProvider

logger = logging.getLogger(__name__)

TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class ElevenLabsProvider(LLMProvider):
    """Answer generation via ElevenLabs Agents (hosted LLM).

    Stage 1 uses a one-shot text simulation:
    - Ensures a text-only agent exists (created once, id reused)
    - Overrides the agent prompt with the grounded RAG prompt
    - Sends the user question as the simulated user's first message
    - Returns the agent's first reply

    STT is a separate provider and is not implemented here.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gemini-2.0-flash",
        api_base: str = "https://api.elevenlabs.io",
        agent_id: str = "",
        timeout_s: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        if not api_key:
            raise GenerationError("ELEVENLABS_API_KEY is not configured", status_code=500)
        self.api_key = api_key
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.agent_id = agent_id
        self.timeout_s = timeout_s
        self.max_retries = max(1, max_retries)
        self._client = httpx.Client(timeout=timeout_s)

    def close(self) -> None:
        self._client.close()

    def _headers(self) -> dict[str, str]:
        return {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, json: dict[str, Any] | None = None) -> httpx.Response:
        url = f"{self.api_base}{path}"
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.request(method, url, headers=self._headers(), json=json)
            except httpx.TimeoutException as exc:
                last_error = exc
                logger.warning("ElevenLabs timeout", extra={"attempt": attempt, "path": path})
                if attempt >= self.max_retries:
                    raise GenerationError("ElevenLabs request timed out", status_code=504) from exc
                time.sleep(min(2 ** (attempt - 1), 8))
                continue
            except httpx.HTTPError as exc:
                last_error = exc
                logger.warning("ElevenLabs HTTP error", extra={"attempt": attempt, "error": str(exc)})
                if attempt >= self.max_retries:
                    raise GenerationError(f"ElevenLabs request failed: {exc}") from exc
                time.sleep(min(2 ** (attempt - 1), 8))
                continue

            if response.status_code in TRANSIENT_STATUS:
                logger.warning(
                    "ElevenLabs transient status",
                    extra={"status": response.status_code, "attempt": attempt},
                )
                if attempt >= self.max_retries:
                    raise GenerationError(
                        f"ElevenLabs returned {response.status_code}",
                        status_code=response.status_code,
                    )
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else min(2 ** (attempt - 1), 8)
                time.sleep(delay)
                continue

            if response.status_code >= 400:
                raise GenerationError(
                    f"ElevenLabs API error {response.status_code}: {response.text[:300]}",
                    status_code=response.status_code,
                )
            return response
        raise GenerationError(f"ElevenLabs request failed: {last_error}")

    def ensure_agent(self) -> str:
        if self.agent_id:
            return self.agent_id
        payload = {
            "name": "hh-goa-rag-stage1",
            "conversation_config": {
                "agent": {
                    "prompt": {
                        "prompt": "Answer using only the provided context.",
                        "llm": self.model,
                    },
                    "first_message": "",
                    "language": "en",
                },
                "conversation": {"text_only": True},
            },
        }
        response = self._request("POST", "/v1/convai/agents/create", json=payload)
        data = response.json()
        agent_id = data.get("agent_id") or data.get("id")
        if not agent_id:
            raise GenerationError(f"ElevenLabs agent create returned no id: {data}")
        self.agent_id = str(agent_id)
        logger.info("Created ElevenLabs agent", extra={"agent_id": self.agent_id})
        return self.agent_id

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> GenerationResult:
        """`prompt` is the user-facing question+context block; system_prompt is optional extra policy.

        The full grounded prompt is sent as the simulated user message so retrieved
        context cannot be dropped by agent override settings.
        """
        agent_id = self.ensure_agent()
        full_user_message = prompt if not system_prompt else f"{system_prompt}\n\n{prompt}"
        agent_system = (
            "You are a retrieval-grounded assistant. "
            "The user message contains the question and retrieved context. "
            "Answer ONLY from that context. If the context is insufficient, refuse."
        )
        payload = {
            "simulation_specification": {
                "simulated_user_config": {
                    "first_message": full_user_message,
                    "language": "en",
                    "prompt": {
                        "prompt": (
                            "Send exactly the message you were given as your first message. "
                            "Do not rewrite, shorten, or omit the retrieved context."
                        ),
                        "llm": self.model,
                    },
                }
            },
            "conversation_config_override": {
                "agent": {
                    "prompt": {
                        "prompt": agent_system,
                        "llm": self.model,
                    },
                    "first_message": "",
                    "language": "en",
                },
                "conversation": {"text_only": True},
            },
            "new_turns_limit": 1,
        }
        t0 = time.perf_counter()
        response = self._request("POST", f"/v1/convai/agents/{agent_id}/simulate-conversation", json=payload)
        latency_ms = (time.perf_counter() - t0) * 1000
        data = response.json()
        text = _extract_agent_text(data)
        if not text:
            raise GenerationError("ElevenLabs returned an empty generation")
        return GenerationResult(
            text=text,
            latency_ms=latency_ms,
            model=self.model,
            raw={"status": response.status_code},
        )


def _extract_agent_text(data: dict[str, Any]) -> str:
    simulated = data.get("simulated_conversation") or data.get("conversation") or []
    if isinstance(simulated, list):
        for turn in reversed(simulated):
            role = str(turn.get("role") or turn.get("source") or "").lower()
            message = turn.get("message") or turn.get("text") or ""
            if isinstance(message, dict):
                message = message.get("text") or ""
            if role in {"agent", "assistant"} and message:
                return str(message).strip()
        for turn in reversed(simulated):
            message = turn.get("message") or turn.get("text") or ""
            if message:
                return str(message).strip()
    if isinstance(data.get("analysis"), dict):
        transcript = data["analysis"].get("transcript")
        if isinstance(transcript, str) and transcript.strip():
            return transcript.strip()
    return ""
