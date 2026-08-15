import httpx
import pytest

from app.core.exceptions import STTError
from rag.stt.elevenlabs import ElevenLabsSTTProvider


def test_stt_posts_official_multipart_contract(monkeypatch) -> None:
    provider = ElevenLabsSTTProvider("test-key", max_retries=1)
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return httpx.Response(200, json={"text": "What is a corporation?", "language_code": "en"})

    monkeypatch.setattr(provider._client, "post", fake_post)
    result = provider.transcribe(b"audio", filename="question.wav", content_type="audio/wav")
    assert result.text == "What is a corporation?"
    assert result.language == "en"
    assert captured["url"].endswith("/v1/speech-to-text")
    assert captured["data"] == {"model_id": "scribe_v2"}
    assert captured["files"]["file"][0] == "question.wav"


def test_stt_transient_failure_is_controlled(monkeypatch) -> None:
    provider = ElevenLabsSTTProvider("test-key", max_retries=1)
    monkeypatch.setattr(
        provider._client,
        "post",
        lambda *args, **kwargs: httpx.Response(429, json={"detail": "rate limit"}),
    )
    with pytest.raises(STTError) as exc_info:
        provider.transcribe(b"audio", filename="question.wav", content_type="audio/wav")
    assert exc_info.value.status_code == 429
