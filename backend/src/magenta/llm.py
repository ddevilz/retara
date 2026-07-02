"""Provider-agnostic LLM client. Plain openai + LangSmith wrap_openai. NO LangChain/LiteLLM.

Roles map to models.yaml: cheap->CHEAP, large->LARGE, judge->JUDGE.
base_url (env OPENAI_BASE_URL, honored by the openai SDK) keeps providers swappable.
Set LANGSMITH_TRACING=true + LANGSMITH_API_KEY to trace; wrap_openai is a no-op otherwise.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

import openai
from langsmith.wrappers import wrap_openai
from pydantic import BaseModel

from magenta.config import load_models

_ROLE_TO_KEY = {"cheap": "CHEAP", "large": "LARGE", "judge": "JUDGE"}

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


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
    return load_models()[key]


def chat(role: str, messages: list[dict], **kw) -> str:
    """Single chat completion, returns assistant text content."""
    client = get_client()
    resp = client.chat.completions.create(
        model=_model_for(role), messages=messages, **kw
    )
    return resp.choices[0].message.content or ""


def chat_structured(
    role: str, messages: list[dict], model_cls: type[BaseModel]
) -> BaseModel:
    """Structured output -> Pydantic instance.

    Tries native structured parse (json_schema); falls back to JSON mode +
    Pydantic validation because json_schema support varies across Groq models.
    """
    client = get_client()
    try:
        resp = client.chat.completions.parse(
            model=_model_for(role), messages=messages, response_format=model_cls
        )
        return resp.choices[0].message.parsed
    except Exception:
        schema_hint = {
            "role": "system",
            "content": "Reply ONLY with a JSON object matching this schema: "
                       + json.dumps(model_cls.model_json_schema()),
        }
        resp = client.chat.completions.create(
            model=_model_for(role), messages=[schema_hint, *messages],
            response_format={"type": "json_object"},
        )
        return model_cls.model_validate_json(
            resp.choices[0].message.content or "{}")
