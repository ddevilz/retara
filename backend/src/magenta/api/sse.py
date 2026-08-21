"""SSE serialization helpers.

sse-starlette's EventSourceResponse expects an async generator yielding dicts
shaped like {"event": <name>, "data": <str>}. We JSON-encode payloads with a
fallback encoder so pydantic models / enums / dataclasses survive the trip.
"""
from __future__ import annotations

import dataclasses
import enum
import json
from collections.abc import AsyncIterator
from typing import Any

from magenta.logging_config import get_logger

logger = get_logger(__name__)


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


async def guarded_stream(gen: AsyncIterator[dict], context: str) -> AsyncIterator[dict]:
    """Convert any exception raised mid-stream into a terminal error event.

    Without this a failure kills the connection and the client sees a truncated stream
    with no explanation. The exception detail is logged, never sent: it can contain
    customer identifiers, and an SSE payload is client-visible.
    """
    try:
        async for event in gen:
            yield event
    except Exception:
        logger.exception("sse.stream_failed", context=context)
        yield sse_event("error", {"message": "the run failed; please retry"})
        yield sse_event("done", {})
