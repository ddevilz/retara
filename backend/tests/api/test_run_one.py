import json
from dataclasses import dataclass

import pytest

from magenta.api import routes_stream as rs


class _FakeGraph:
    def stream(self, initial, config=None, stream_mode=None):
        # Emit as (mode, {node: payload}) to match stream_mode=["updates"].
        # `config` accepted-and-ignored: the real graph.stream() call always
        # passes a {"configurable": {"thread_id": ...}} config (required once
        # a checkpointer is set — see routes_stream.py docstring), so the
        # fake's signature has to tolerate it too.
        yield ("updates", {"brain": {"segment": "persuadable", "p_churn": 0.61}})
        yield ("updates", {"diagnose": {"root_cause_tags": ["bill_shock"]}})
        yield ("updates", {"nba": {"offer_id": "bill_credit_or_waiver"}})
        yield ("updates", {"guardrail": {"verdict": "pass"}})
        yield ("updates", {"act": {"status": "FULFILLED"}})


def _parse_sse(text: str):
    events = []
    cur = {}
    for line in text.splitlines():
        if line.startswith("event:"):
            cur["event"] = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            cur["data"] = line.split(":", 1)[1].strip()
        elif line == "" and cur:
            events.append(cur)
            cur = {}
    if cur:
        events.append(cur)
    return events


@pytest.fixture
def patched_graph(monkeypatch):
    @dataclass
    class _FakeDeps:
        tenant_id: str = "org_default"

    monkeypatch.setattr(rs, "build_graph", lambda deps: _FakeGraph())
    monkeypatch.setattr(rs, "get_graph_deps", lambda: _FakeDeps())

    class _C:
        customer_id = "CUST-DEMO"
    monkeypatch.setattr(rs, "_find_customer", lambda cid: _C())


@pytest.mark.asyncio
async def test_run_one_streams_node_events(client, patched_graph):
    resp = await client.post("/api/run-one", json={"customer_id": "CUST-DEMO"})
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    node_events = [e for e in events if e["event"] == "node"]
    assert len(node_events) >= 3
    first = json.loads(node_events[0]["data"])
    assert "node" in first and "payload" in first
    assert events[-1]["event"] == "done"


@pytest.mark.asyncio
async def test_run_one_unknown_customer(client, monkeypatch):
    monkeypatch.setattr(rs, "_find_customer", lambda cid: None)
    resp = await client.post("/api/run-one", json={"customer_id": "NOPE-99999"})
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    assert events[0]["event"] == "error"
    assert events[-1]["event"] == "done"
