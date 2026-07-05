"""Provider-agnostic LLM client. Plain openai + LangSmith wrap_openai. NO LangChain/LiteLLM.

Roles map to models.yaml: cheap->CHEAP, large->LARGE, judge->JUDGE.
base_url (env OPENAI_BASE_URL, honored by the openai SDK) keeps providers swappable.
Set LANGSMITH_TRACING=true + LANGSMITH_API_KEY to trace; wrap_openai is a no-op otherwise.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from functools import lru_cache

import openai
from langsmith.wrappers import wrap_openai
from pydantic import BaseModel, ValidationError

from magenta.config import load_models

_ROLE_TO_KEY = {"cheap": "CHEAP", "large": "LARGE", "judge": "JUDGE"}

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

## --------------------------------------------------------------------------- #
## 429 rate-limit backoff. Groq's free tier throws openai.RateLimitError with a
## retry-after hint baked into the message (e.g. "Please try again in
## 1m21.216s") when TPM/TPD caps hit. A single 429 must never kill an
## unattended multi-hour cohort run: retry with the provider's own hint,
## bounded so a stuck cap can't hang forever -- once bounds are exhausted the
## error is re-raised so callers (e.g. System-2, see graph/nodes.py::decide)
## can degrade instead of dying.
## --------------------------------------------------------------------------- #
RATE_LIMIT_MAX_ATTEMPTS = 5       # total call attempts (initial + retries)
RATE_LIMIT_ATTEMPT_SLEEP_CAP_S = 300.0   # never sleep longer than this per attempt
RATE_LIMIT_TOTAL_BUDGET_S = 900.0        # never sleep more than this in total

_RETRY_AFTER_MIN_SEC_RE = re.compile(r"in\s+(\d+)m(\d+(?:\.\d+)?)s", re.IGNORECASE)
_RETRY_AFTER_SEC_RE = re.compile(r"in\s+(\d+(?:\.\d+)?)s", re.IGNORECASE)


def _parse_retry_after_seconds(message: str) -> float | None:
    """Extract a retry-after duration from a rate-limit error message.

    Handles both Groq-style forms: "...in 1m21.216s" (minutes+seconds) and
    "...in 21.216s" (seconds only). Returns None if no hint is found, in
    which case the caller falls back to the per-attempt sleep cap.
    """
    m = _RETRY_AFTER_MIN_SEC_RE.search(message)
    if m:
        minutes, seconds = m.groups()
        return float(minutes) * 60.0 + float(seconds)
    m = _RETRY_AFTER_SEC_RE.search(message)
    if m:
        return float(m.group(1))
    return None


def _call_with_retry(fn, *args, **kw):
    """Invoke an OpenAI-compatible call, retrying on 429 with the provider's
    own retry-after hint. Bounded by RATE_LIMIT_MAX_ATTEMPTS and
    RATE_LIMIT_TOTAL_BUDGET_S; each sleep is capped at
    RATE_LIMIT_ATTEMPT_SLEEP_CAP_S. Logs every wait to stderr so run logs show
    stalls. Re-raises the RateLimitError once bounds are exhausted.
    """
    total_slept = 0.0
    for attempt in range(1, RATE_LIMIT_MAX_ATTEMPTS + 1):
        try:
            return fn(*args, **kw)
        except openai.RateLimitError as exc:
            if attempt >= RATE_LIMIT_MAX_ATTEMPTS:
                raise
            wait = _parse_retry_after_seconds(str(exc))
            if wait is None:
                wait = RATE_LIMIT_ATTEMPT_SLEEP_CAP_S
            wait = min(wait, RATE_LIMIT_ATTEMPT_SLEEP_CAP_S)
            if total_slept + wait > RATE_LIMIT_TOTAL_BUDGET_S:
                raise
            print(
                f"[magenta.llm] 429 rate limited (attempt {attempt}/"
                f"{RATE_LIMIT_MAX_ATTEMPTS}); sleeping {wait:.1f}s before retry: {exc}",
                file=sys.stderr,
            )
            time.sleep(wait)
            total_slept += wait
    raise AssertionError("unreachable: loop above always returns or raises")


@lru_cache(maxsize=1)
def get_client():
    """wrap_openai(openai.OpenAI()) — cached.

    Default provider = Groq: when GROQ_API_KEY is set (and OPENAI_API_KEY is not),
    the client targets Groq's OpenAI-compatible endpoint. Explicit
    OPENAI_API_KEY / OPENAI_BASE_URL always win — any compatible provider swaps in.
    """
    if os.environ.get("GROQ_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        return wrap_openai(openai.OpenAI(
            api_key=os.environ["GROQ_API_KEY"], base_url=GROQ_BASE_URL))
    return wrap_openai(openai.OpenAI())


def _model_for(role: str) -> str:
    key = _ROLE_TO_KEY.get(role)
    if key is None:
        raise ValueError(f"unknown role {role!r}; expected one of {sorted(_ROLE_TO_KEY)}")
    # Env override (e.g. MAGENTA_MODEL_LARGE=llama-3.1-8b-instant): lets long
    # cohort runs dodge per-model daily token caps on free tiers without
    # touching the committed models.yaml. Documented in run logs when used.
    override = os.environ.get(f"MAGENTA_MODEL_{key}")
    return override or load_models()[key]


def chat(role: str, messages: list[dict], **kw) -> str:
    """Single chat completion, returns assistant text content.

    A 429 is retried with bounded backoff (see _call_with_retry); once
    retries are exhausted the RateLimitError propagates to the caller.
    """
    client = get_client()
    resp = _call_with_retry(
        client.chat.completions.create, model=_model_for(role), messages=messages, **kw
    )
    return resp.choices[0].message.content or ""


def chat_structured(
    role: str, messages: list[dict], model_cls: type[BaseModel]
) -> BaseModel:
    """Structured output -> Pydantic instance.

    Tries native structured parse (json_schema); falls back to JSON mode +
    Pydantic validation because json_schema support varies across Groq models.
    Each underlying network call is retried with bounded backoff on 429 (see
    _call_with_retry). A RateLimitError that survives those retries is
    re-raised immediately -- it is a capacity problem, not a
    schema-unsupported problem, so it must NOT fall through to the JSON-mode
    fallback (which would just burn the same exhausted budget again).
    """
    client = get_client()
    try:
        resp = _call_with_retry(
            client.chat.completions.parse,
            model=_model_for(role), messages=messages, response_format=model_cls,
        )
        return resp.choices[0].message.parsed
    except openai.RateLimitError:
        raise
    except Exception:
        schema_hint = {
            "role": "system",
            "content": "Reply ONLY with a JSON object matching this schema: "
                       + json.dumps(model_cls.model_json_schema()),
        }
        resp = _call_with_retry(
            client.chat.completions.create,
            model=_model_for(role), messages=[schema_hint, *messages],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        try:
            return model_cls.model_validate_json(raw)
        except ValidationError as exc:
            # One self-repair retry: feed the validation error back. A single
            # malformed reply must never kill a cohort run (a missing
            # eligible_offer_ids crashed a live 5-rung ablation).
            repair = {
                "role": "system",
                "content": ("Your previous JSON failed validation: "
                            f"{exc.errors(include_url=False)}. Reply ONLY with a "
                            "corrected JSON object matching the schema."),
            }
            resp = _call_with_retry(
                client.chat.completions.create,
                model=_model_for(role), messages=[schema_hint, *messages, repair],
                response_format={"type": "json_object"},
            )
            return model_cls.model_validate_json(
                resp.choices[0].message.content or "{}")
