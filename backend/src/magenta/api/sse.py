"""SSE serialization helpers.

sse-starlette's EventSourceResponse expects an async generator yielding dicts
shaped like {"event": <name>, "data": <str>}. We JSON-encode payloads with a
fallback encoder so pydantic models / enums / dataclasses survive the trip.
"""
from __future__ import annotations

import dataclasses
import enum
import json
from typing import Any


def _default(o: Any) -> Any:
    if isinstance(o, enum.Enum):
        return o.value
    if hasattr(o, "model_dump"):          # pydantic v2
        return o.model_dump()
    if hasattr(o, "dict"):                # pydantic v1 fallback
        return o.dict()
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.asdict(o)
    if isinstance(o, (set, frozenset)):
        return list(o)
    return str(o)                          # last resort — never crash a stream


def to_json(payload: Any) -> str:
    return json.dumps(payload, default=_default)


def sse_event(event: str, payload: Any) -> dict:
    return {"event": event, "data": to_json(payload)}
