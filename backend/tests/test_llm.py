import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

import magenta.llm as llm


class _Diag(BaseModel):
    root_cause: str
    confidence: float


def _fake_completion(text: str):
    msg = SimpleNamespace(content=text, parsed=None)
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


def _fake_parsed(model: BaseModel):
    msg = SimpleNamespace(content=None, parsed=model)
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


class FakeClient:
    """Stand-in for a wrap_openai(OpenAI()) client. Records calls."""

    def __init__(self):
        self.create_calls = []
        self.parse_calls = []
        completions = SimpleNamespace(create=self._create, parse=self._parse)
        self.chat = SimpleNamespace(completions=completions)

    def _create(self, *, model, messages, **kw):
        self.create_calls.append({"model": model, "messages": messages, "kw": kw})
        return _fake_completion("hello from fake")

    def _parse(self, *, model, messages, response_format, **kw):
        self.parse_calls.append({"model": model, "response_format": response_format})
        return _fake_parsed(_Diag(root_cause="BILL_SHOCK", confidence=0.9))


@pytest.fixture
def fake(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(llm, "get_client", lambda: client)
    return client


def test_role_maps_to_model_id(fake):
    out = llm.chat("cheap", [{"role": "user", "content": "hi"}])
    assert out == "hello from fake"
    assert fake.create_calls[0]["model"] == "llama-3.1-8b-instant"


def test_large_and_judge_roles(fake):
    llm.chat("large", [{"role": "user", "content": "x"}])
    llm.chat("judge", [{"role": "user", "content": "y"}])
    assert fake.create_calls[0]["model"] == "llama-3.3-70b-versatile"
    assert fake.create_calls[1]["model"] == "openai/gpt-oss-120b"


def test_unknown_role_raises(fake):
    with pytest.raises(ValueError):
        llm.chat("wizard", [{"role": "user", "content": "x"}])


def test_kwargs_forwarded(fake):
    llm.chat("cheap", [{"role": "user", "content": "x"}], temperature=0.0, seed=7)
    kw = fake.create_calls[0]["kw"]
    assert kw["temperature"] == 0.0
    assert kw["seed"] == 7


def test_chat_structured_returns_model(fake):
    out = llm.chat_structured("large", [{"role": "user", "content": "x"}], _Diag)
    assert isinstance(out, _Diag)
    assert out.root_cause == "BILL_SHOCK"
    assert fake.parse_calls[0]["model"] == "llama-3.3-70b-versatile"
    assert fake.parse_calls[0]["response_format"] is _Diag


class FakeClientParseUnsupported:
    """Stand-in for a provider (e.g. a Groq model) whose `.parse` endpoint
    doesn't support structured outputs — raises, forcing the JSON-mode
    fallback in `chat_structured`."""

    def __init__(self, content: str):
        self._content = content
        self.create_calls = []
        completions = SimpleNamespace(create=self._create, parse=self._parse)
        self.chat = SimpleNamespace(completions=completions)

    def _parse(self, *, model, messages, response_format, **kw):
        raise RuntimeError("this model does not support response_format=json_schema")

    def _create(self, *, model, messages, **kw):
        self.create_calls.append({"model": model, "messages": messages, "kw": kw})
        return _fake_completion(self._content)


def test_chat_structured_falls_back_to_json_mode_when_parse_unsupported(monkeypatch):
    content = json.dumps({"root_cause": "NETWORK_OUTAGE", "confidence": 0.42})
    client = FakeClientParseUnsupported(content)
    monkeypatch.setattr(llm, "get_client", lambda: client)

    out = llm.chat_structured("large", [{"role": "user", "content": "x"}], _Diag)

    assert isinstance(out, _Diag)
    assert out.root_cause == "NETWORK_OUTAGE"
    assert out.confidence == 0.42

    assert len(client.create_calls) == 1
    fallback_call = client.create_calls[0]
    assert fallback_call["model"] == "llama-3.3-70b-versatile"
    assert fallback_call["kw"]["response_format"] == {"type": "json_object"}


@pytest.fixture
def clear_get_client_cache():
    """get_client is @lru_cache'd — clear before and after so cached state
    from one test never leaks into the next."""
    llm.get_client.cache_clear()
    yield
    llm.get_client.cache_clear()


class RecordingOpenAI:
    """Fake stand-in for openai.OpenAI — records constructor kwargs instead
    of touching the network."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_get_client_uses_groq_when_only_groq_key_set(
    monkeypatch, clear_get_client_cache
):
    recorded_clients = []

    def fake_openai(**kwargs):
        client = RecordingOpenAI(**kwargs)
        recorded_clients.append(client)
        return client

    monkeypatch.setattr(llm.openai, "OpenAI", fake_openai)
    monkeypatch.setattr(llm, "wrap_openai", lambda client: client)
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = llm.get_client()

    assert client.kwargs["base_url"] == "https://api.groq.com/openai/v1"
    assert client.kwargs["api_key"] == "fake-groq"
    assert len(recorded_clients) == 1


def test_get_client_prefers_openai_when_both_keys_set(
    monkeypatch, clear_get_client_cache
):
    recorded_clients = []

    def fake_openai(**kwargs):
        client = RecordingOpenAI(**kwargs)
        recorded_clients.append(client)
        return client

    monkeypatch.setattr(llm.openai, "OpenAI", fake_openai)
    monkeypatch.setattr(llm, "wrap_openai", lambda client: client)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai")
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq")

    client = llm.get_client()

    assert "base_url" not in client.kwargs
    assert client.kwargs == {}
    assert len(recorded_clients) == 1
