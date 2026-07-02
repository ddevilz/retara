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
