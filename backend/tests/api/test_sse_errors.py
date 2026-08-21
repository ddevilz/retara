import json

import pytest

from magenta.api.sse import guarded_stream, sse_event


async def _boom():
    yield sse_event("progress", {"pct": 10})
    raise RuntimeError("customer CUST_0001 secret detail")


@pytest.mark.asyncio
async def test_exception_becomes_an_error_then_done_event():
    events = [ev async for ev in guarded_stream(_boom(), context="test")]
    assert events[0]["event"] == "progress"
    assert events[1]["event"] == "error"
    assert events[2]["event"] == "done"


@pytest.mark.asyncio
async def test_error_event_does_not_leak_exception_detail():
    """Exception text can contain customer identifiers. The client gets a generic
    message; the detail goes to the log."""
    events = [ev async for ev in guarded_stream(_boom(), context="test")]
    payload = json.loads(events[1]["data"])
    assert "CUST_0001" not in json.dumps(payload)
    assert "secret detail" not in json.dumps(payload)
