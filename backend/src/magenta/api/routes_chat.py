"""Chat endpoints: start a session, stream a per-turn ChatReply.

Brief bugs fixed on sight (beyond the ones called out in the task):

1. `_build_chat`: the brief guessed a `deps.risk.assess(customer)` method
   gated behind `hasattr(...)` and passed `diagnosis=None` when that guess
   inevitably missed (`RiskModel` only exposes `.score()` — grepped
   `magenta/brain/risk.py`). `RetentionChat._customer_360` unconditionally
   reads `report.drivers` and `diagnosis.narrative`, so the very first turn
   would crash with `AttributeError`. Route through the same `sense`/
   `diagnose` node functions `magenta.chat.runner._build_context` already
   uses, so this endpoint doesn't fork its own (broken) copy of that scoring
   logic — mirrors the cli `chat` command's wiring via `run_negotiation`.
2. `_pick_customer` (Phase 1.3 Task 3): now reads the per-tenant population
   from `magenta.api.population.get_population(tenant_id)` instead of its own
   module-level `@lru_cache(maxsize=1)` demo population — that cache and
   `data_access`'s and `deps`'s identically-seeded copies were the same bug
   three times (CLAUDE.md, Phase 1.3 build ledger). `deps.py` (Task 4) now
   also resolves through `get_population(tenant_id)`, so `_build_chat` below
   gets a Customer whose scored attributes match this tenant's population.
3. Persona construction used a function-level `from magenta.chat.persona
   import Archetype, make_persona` — this repo's hard rule is "all imports
   at module top, no function-level imports ever" (CLAUDE.md). Hoisted.
4. The session's local `RetentionChat` variable was named `chat` in the
   brief, shadowing this module's own subject matter (`magenta.chat.*`) the
   same way `magenta.chat.agent._wording` had to rename its `act` parameter
   to avoid shadowing the imported `act` graph node. Named `chat_agent`
   instead, matching the name `magenta.chat.runner.run_negotiation` already
   uses for the same object.
5. `chat_sessions.ChatSession.persona` was annotated as
   `magenta.chat.persona.PersonaAgent` but `make_persona(...)` returns a
   `PersonaBrief` (`PersonaAgent` wraps a brief and drives the *persona's*
   side of a scripted negotiation — not needed here, since a turn's `text`
   is supplied by the caller regardless of mode). Fixed in chat_sessions.py.
"""
from __future__ import annotations

from typing import cast

import anyio
from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from magenta.api import chat_sessions as cs
from magenta.api.deps import get_graph_deps
from magenta.api.population import get_population
from magenta.api.schemas import ChatStartRequest, ChatStartResponse, ChatTurnRequest
from magenta.api.sse import sse_event
from magenta.auth import TenantContext, current_tenant
from magenta.chat.agent import RetentionChat
from magenta.chat.persona import Archetype, make_persona
from magenta.graph import diagnose, sense
from magenta.graph.state import OverallState

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _pick_customer(tenant_id: str, customer_id: str | None):
    population = get_population(tenant_id)
    if customer_id:
        return population.customers.get(customer_id)
    return next(iter(population.customers.values()))  # default demo customer


def _build_chat(customer, tenant_id: str):
    """Build a RetentionChat with a *real* report+diagnosis, driving the same
    sense/diagnose node functions the decision graph and negotiation runner
    use (mirrors `magenta.chat.runner._build_context`) instead of forking a
    second, broken copy of that scoring logic. `sense`/`diagnose` re-load the
    customer via `deps.load_customer(customer_id)`, which now (Task 4) is
    this same tenant's own population, so the reloaded Customer's attribute
    values match `customer` above.

    `get_graph_deps(tenant_id)` returns this tenant's own cached GraphDeps
    (`deps.tenant_id` already set correctly) — the session's RetentionChat
    later reaches `act()` on this deps object via `chat/agent.py`, which must
    not write FULFILLMENTS/GUARDRAIL_CONTACTS under the wrong tenant. May
    raise `ModelsNotReady` -> 503 (this call happens outside any SSE
    generator, in `chat_start`, so the exception handler sees it)."""
    deps = get_graph_deps(tenant_id)
    # Deliberately partial: same incremental sense()/diagnose() drive as
    # magenta.chat.runner._build_context -- each call only reads keys the
    # prior call already populated.
    state: dict = {"customer_id": customer.customer_id}
    state.update(sense(cast(OverallState, state), deps))
    state.update(diagnose(cast(OverallState, state), deps))
    return RetentionChat(deps, customer, state["risk"], state["diagnosis"], authority_cap=80.0)


@router.post("/start", response_model=ChatStartResponse)
def chat_start(
    req: ChatStartRequest,
    tenant: TenantContext = Depends(current_tenant),
) -> ChatStartResponse:
    if req.mode == "persona" and not req.archetype:
        raise HTTPException(422, "archetype required for persona mode")
    customer = _pick_customer(tenant.tenant_id, req.customer_id)
    if customer is None:
        raise HTTPException(404, f"unknown customer {req.customer_id}")

    chat_agent = _build_chat(customer, tenant.tenant_id)

    persona = None
    if req.mode == "persona":
        assert req.archetype is not None  # guaranteed by the mode/archetype check above
        try:
            arche = Archetype[req.archetype] if req.archetype in Archetype.__members__ \
                else Archetype(req.archetype)
        except ValueError:
            raise HTTPException(422, f"unknown archetype {req.archetype}") from None
        hidden = get_population(tenant.tenant_id).hidden
        hidden_state = hidden.get(customer.customer_id)
        if hidden_state is None:
            # HiddenStore is generated alongside the population it indexes, so a
            # customer resolved from that same population always has an entry;
            # a miss means the two are out of sync, not something to paper over.
            raise HTTPException(500, f"no hidden state for {customer.customer_id}")
        persona = make_persona(arche, customer, hidden_state)

    sid = cs.new_id()
    cs.create(cs.ChatSession(
        session_id=sid,
        tenant_id=tenant.tenant_id,
        mode=req.mode,
        customer_id=getattr(customer, "customer_id", "CUST-DEMO"),
        archetype=req.archetype,
        chat=chat_agent,
        persona=persona,
    ))
    return ChatStartResponse(
        session_id=sid, mode=req.mode,
        customer_id=getattr(customer, "customer_id", "CUST-DEMO"),
        archetype=req.archetype,
    )


@router.post("/{session_id}/turn")
async def chat_turn(
    session_id: str,
    req: ChatTurnRequest,
    tenant: TenantContext = Depends(current_tenant),
):
    session = cs.get(session_id, tenant.tenant_id)
    if session is None:
        raise HTTPException(404, f"unknown session {session_id}")

    async def gen():
        session.history.append({"role": "user", "text": req.text})

        def _respond():
            return session.chat.respond(req.text)

        reply = await anyio.to_thread.run_sync(_respond)  # ChatReply

        # ChatReply(text, act, offer, state) — serialize via sse.to_json fallback.
        payload = {
            "text": getattr(reply, "text", ""),
            "act": getattr(reply, "act", None),
            "offer": getattr(reply, "offer", None),
            "state": getattr(reply, "state", None),
        }
        session.history.append({"role": "agent", "text": payload["text"]})
        yield sse_event("reply", payload)
        # Surface terminal status if present on the dialogue state.
        status = None
        state = payload["state"]
        if isinstance(state, dict):
            status = state.get("status")
        elif state is not None:
            status = getattr(state, "status", None)
        yield sse_event("done", {"status": status})

    return EventSourceResponse(gen())
