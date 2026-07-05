import json
from types import SimpleNamespace

import httpx
import openai
import pytest
from pydantic import BaseModel

import magenta.llm as llm


def _rate_limit_error(message: str) -> openai.RateLimitError:
    """Build a real openai.RateLimitError the way the SDK would on a 429,
    carrying a Groq-style retry-after hint in the message."""
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status_code=429, request=request)
    return openai.RateLimitError(message, response=response, body=None)


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


## --------------------------------------------------------------------------- #
## 429 rate-limit backoff — bounded retry-after handling (unattended cohort runs)
## --------------------------------------------------------------------------- #
class TestParseRetryAfterSeconds:
    def test_plain_seconds_form(self):
        msg = "Rate limit reached. Please try again in 21.216s."
        assert llm._parse_retry_after_seconds(msg) == pytest.approx(21.216)

    def test_minutes_and_seconds_form(self):
        msg = "Rate limit reached. Please try again in 1m21.216s."
        assert llm._parse_retry_after_seconds(msg) == pytest.approx(81.216)

    def test_no_hint_returns_none(self):
        assert llm._parse_retry_after_seconds("nope, no hint here") is None


class FlakyRateLimitedClient:
    """Fake client whose `.create` raises RateLimitError a fixed number of
    times (with a retry-after hint) before succeeding."""

    def __init__(self, fail_times: int, retry_after_msg: str = "try again in 0.01s"):
        self._fail_times = fail_times
        self._retry_after_msg = retry_after_msg
        self.create_calls = 0
        completions = SimpleNamespace(create=self._create, parse=self._parse)
        self.chat = SimpleNamespace(completions=completions)

    def _create(self, *, model, messages, **kw):
        self.create_calls += 1
        if self.create_calls <= self._fail_times:
            raise _rate_limit_error(self._retry_after_msg)
        return _fake_completion("recovered")

    def _parse(self, *, model, messages, response_format, **kw):
        self.create_calls += 1
        if self.create_calls <= self._fail_times:
            raise _rate_limit_error(self._retry_after_msg)
        return _fake_parsed(_Diag(root_cause="RECOVERED", confidence=0.5))


@pytest.fixture
def fake_sleep(monkeypatch):
    """Replace time.sleep with a recorder — never actually sleep in tests."""
    calls = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: calls.append(s))
    return calls


def test_chat_retries_after_429_then_succeeds(monkeypatch, fake_sleep):
    client = FlakyRateLimitedClient(fail_times=1, retry_after_msg="try again in 0.01s")
    monkeypatch.setattr(llm, "get_client", lambda: client)

    out = llm.chat("cheap", [{"role": "user", "content": "hi"}])

    assert out == "recovered"
    assert client.create_calls == 2
    assert fake_sleep == [pytest.approx(0.01)]


def test_chat_retries_exhausted_raises(monkeypatch, fake_sleep):
    # Always 429s -> retries should exhaust at RATE_LIMIT_MAX_ATTEMPTS and re-raise.
    client = FlakyRateLimitedClient(fail_times=999, retry_after_msg="try again in 0.01s")
    monkeypatch.setattr(llm, "get_client", lambda: client)

    with pytest.raises(openai.RateLimitError):
        llm.chat("cheap", [{"role": "user", "content": "hi"}])

    assert client.create_calls == llm.RATE_LIMIT_MAX_ATTEMPTS
    assert len(fake_sleep) == llm.RATE_LIMIT_MAX_ATTEMPTS - 1


def test_chat_structured_retries_after_429_then_succeeds(monkeypatch, fake_sleep):
    client = FlakyRateLimitedClient(fail_times=1, retry_after_msg="try again in 0.01s")
    monkeypatch.setattr(llm, "get_client", lambda: client)

    out = llm.chat_structured("large", [{"role": "user", "content": "x"}], _Diag)

    assert isinstance(out, _Diag)
    assert out.root_cause == "RECOVERED"
    assert fake_sleep == [pytest.approx(0.01)]


def test_chat_structured_retries_exhausted_raises(monkeypatch, fake_sleep):
    client = FlakyRateLimitedClient(fail_times=999, retry_after_msg="try again in 0.01s")
    monkeypatch.setattr(llm, "get_client", lambda: client)

    with pytest.raises(openai.RateLimitError):
        llm.chat_structured("large", [{"role": "user", "content": "x"}], _Diag)

    assert len(fake_sleep) == llm.RATE_LIMIT_MAX_ATTEMPTS - 1


def test_sleep_is_capped_per_attempt(monkeypatch, fake_sleep):
    # retry-after hint (600s) exceeds the per-attempt cap -> sleep is clamped.
    client = FlakyRateLimitedClient(fail_times=1, retry_after_msg="try again in 10m0s")
    monkeypatch.setattr(llm, "get_client", lambda: client)

    llm.chat("cheap", [{"role": "user", "content": "hi"}])

    assert fake_sleep == [llm.RATE_LIMIT_ATTEMPT_SLEEP_CAP_S]


def test_total_budget_exhausted_raises_before_max_attempts(monkeypatch, fake_sleep):
    # Each wait is just under the per-attempt cap; the cumulative total budget
    # is exhausted well before RATE_LIMIT_MAX_ATTEMPTS tries are used up.
    big_wait = llm.RATE_LIMIT_ATTEMPT_SLEEP_CAP_S
    client = FlakyRateLimitedClient(fail_times=999, retry_after_msg=f"try again in {big_wait}s")
    monkeypatch.setattr(llm, "get_client", lambda: client)

    with pytest.raises(openai.RateLimitError):
        llm.chat("cheap", [{"role": "user", "content": "hi"}])

    assert client.create_calls < llm.RATE_LIMIT_MAX_ATTEMPTS
    assert sum(fake_sleep) <= llm.RATE_LIMIT_TOTAL_BUDGET_S
