from rag.generation.elevenlabs import ElevenLabsProvider, _extract_agent_text
from rag.generation.prompts import REFUSAL_MESSAGE, build_generation_prompt


def test_prompt_contains_grounding_rules() -> None:
    prompt = build_generation_prompt("What is X?", "[irrelevant documents]")
    assert "Never invent facts" in prompt
    assert "What is X?" in prompt
    assert REFUSAL_MESSAGE.split()[0] in prompt or "couldn't find" in prompt.lower()


def test_extract_agent_text() -> None:
    data = {
        "simulated_conversation": [
            {"role": "user", "message": "What is Paris?"},
            {"role": "agent", "message": "Paris is a city in France."},
        ]
    }
    assert _extract_agent_text(data) == "Paris is a city in France."


def test_elevenlabs_retries_then_succeeds(monkeypatch) -> None:
    calls = {"n": 0}

    class DummyResponse:
        def __init__(self, status_code: int, payload=None):
            self.status_code = status_code
            self._payload = payload or {"agent_id": "abc"}
            self.text = "ok"
            self.headers = {}

        def json(self):
            return self._payload

    class DummyClient:
        def request(self, method, url, headers=None, json=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return DummyResponse(503)
            return DummyResponse(200, {"agent_id": "agent_1"})

        def close(self):
            pass

    provider = ElevenLabsProvider("sk_test", max_retries=3, timeout_s=1)
    provider._client = DummyClient()
    monkeypatch.setattr("time.sleep", lambda *_: None)
    agent_id = provider.ensure_agent()
    assert agent_id == "agent_1"
    assert calls["n"] == 2
