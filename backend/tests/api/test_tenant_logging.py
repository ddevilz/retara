"""Regression test for Important #3 (fix-wave-2): `set_tenant()` must actually reach
log lines emitted while a real authenticated request is being handled, not just work
as a bare function in isolation (that much was already covered by test_logging.py).

`current_tenant` is a sync dependency FastAPI dispatches to a worker thread; binding
the tenant contextvar inside it would be discarded when that thread returns. The fix
is `magenta.auth.bound_tenant`, an `async def` wrapper every route now depends on
instead, which binds on the event loop where the context actually persists into the
route handler. This test proves that end-to-end: it makes a real request through the
ASGI app and asserts a log line emitted from *inside* the (sync) route handler --
`scorecards.served` in routes_data.py -- carries the resolved tenant_id.
"""
from __future__ import annotations

import json

import pytest

from magenta.logging_config import configure_logging
from tests.db_fixtures import TENANT_A


@pytest.mark.asyncio
async def test_log_line_from_route_handler_carries_tenant_id(client, capsys):
    configure_logging()
    resp = await client.get("/api/scorecards")
    assert resp.status_code == 200

    lines = capsys.readouterr().out.strip().splitlines()
    events = [json.loads(line) for line in lines]
    served = [e for e in events if e.get("event") == "scorecards.served"]
    assert served, f"expected a scorecards.served log line, got events: {events}"
    assert served[-1]["tenant_id"] == TENANT_A
