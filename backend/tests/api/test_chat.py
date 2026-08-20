import json

import pytest

from magenta.api import chat_sessions as cs
from magenta.api import routes_chat as rc
from tests.db_fixtures import TENANT_A


class _FakeReply:
    def __init__(self):
        self.text = "I hear you — let's look at a cheaper plan."
        self.act = "NEGOTIATE"
        self.offer = {"arm_id": "plan_downsell_retain", "value_eur": 8.0}
        self.state = {"status": "ACTIVE", "sentiment": -0.2,
                      "ladder_position": 1, "authority_cap": 80.0,
                      "intent_stack": ["price_objection"]}


class _FakeChat:
    def __init__(self, *a, **k):
        self.calls = 0

    def respond(self, text):
        self.calls += 1
        return _FakeReply()


@pytest.fixture(autouse=True)
def _reset_and_patch(monkeypatch):
    cs.clear()
    monkeypatch.setattr(rc, "_build_chat", lambda customer, tenant_id: _FakeChat())

    class _C:
        customer_id = "CUST-DEMO"
    monkeypatch.setattr(rc, "_pick_customer", lambda tenant_id, cid: _C())
    yield
    cs.clear()


def _parse_sse(text: str):
    events, cur = [], {}
    for line in text.splitlines():
        if line.startswith("event:"):
            cur["event"] = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            cur["data"] = line.split(":", 1)[1].strip()
        elif line == "" and cur:
            events.append(cur); cur = {}
    if cur:
        events.append(cur)
    return events


@pytest.mark.asyncio
async def test_chat_lifecycle(client):
    start = await client.post("/api/chat/start",
                              json={"mode": "human", "customer_id": "CUST-DEMO"})
    assert start.status_code == 200
    sid = start.json()["session_id"]
    assert sid.startswith("sess-")

    turn = await client.post(f"/api/chat/{sid}/turn",
                             json={"text": "This bill is way too high"})
    assert turn.status_code == 200
    events = _parse_sse(turn.text)
    replies = [e for e in events if e["event"] == "reply"]
    assert len(replies) == 1
    data = json.loads(replies[0]["data"])
    assert "text" in data and "state" in data
    assert data["state"]["status"] == "ACTIVE"
    assert events[-1]["event"] == "done"


@pytest.mark.asyncio
async def test_chat_unknown_session_404(client):
    resp = await client.post("/api/chat/sess-nope/turn", json={"text": "hi"})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_persona_requires_archetype(client):
    resp = await client.post("/api/chat/start", json={"mode": "persona"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_chat_session_persists_across_two_turns(client):
    """The brief's 'Done' criterion: a session survives across two turn
    requests sharing one session_id (verifies the in-memory registry keeps
    the same RetentionChat instance alive between calls, not a fresh one
    per request)."""
    start = await client.post("/api/chat/start",
                              json={"mode": "human", "customer_id": "CUST-DEMO"})
    sid = start.json()["session_id"]
    session = cs.get(sid, TENANT_A)
    fake_chat = session.chat

    await client.post(f"/api/chat/{sid}/turn", json={"text": "first"})
    await client.post(f"/api/chat/{sid}/turn", json={"text": "second"})

    assert cs.get(sid, TENANT_A).chat is fake_chat
    assert fake_chat.calls == 2


@pytest.mark.asyncio
async def test_persona_mode_unknown_archetype_422(client):
    resp = await client.post("/api/chat/start",
                             json={"mode": "persona", "archetype": "NOT_A_REAL_ARCHETYPE",
                                   "customer_id": "CUST-DEMO"})
    assert resp.status_code == 422
