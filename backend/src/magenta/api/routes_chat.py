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
2. `_pick_customer`: the brief called `generate_population(n, seed)` and
   iterated the result as if it were a bare customer list. It actually
   returns `(list[Customer], HiddenStore)` (see `magenta.sim.population`
   docstring, and the same deviation already called out in
   `magenta.api.data_access`'s module docstring) — iterating the 2-tuple
   directly would walk over the list object and the HiddenStore object, not
   customers. Unpack explicitly and cache it (module-level population, same
   DEMO_POP_N/SEED `data_access`/`deps` use, imported rather than
   re-declared so this can't silently drift onto a different population).
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

from functools import lru_cache

import anyio
from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from magenta.api import chat_sessions as cs
from magenta.api.data_access import DEMO_POP_N, DEMO_POP_SEED
from magenta.api.deps import get_graph_deps
from magenta.api.schemas import ChatStartRequest, ChatStartResponse, ChatTurnRequest
from magenta.api.sse import sse_event
from magenta.auth import TenantContext, current_tenant
from magenta.chat.agent import RetentionChat
from magenta.chat.persona import Archetype, make_persona
from magenta.graph import diagnose, sense
from magenta.sim.population import generate_population

router = APIRouter(prefix="/api/chat", tags=["chat"])


@lru_cache(maxsize=1)
def _demo_population():
    """One generation of the same demo population `data_access`/`deps` serve
    (DEMO_POP_N/SEED imported, not re-declared, so this can't drift). Cached
    module-locally rather than reused from `magenta.api.deps` because that
    module's population cache is private to its own `find_customer` lookup —
    same precedent as `data_access._demo_population` keeping its own cache
    off the identical seed rather than reaching into `deps`."""
    return generate_population(DEMO_POP_N, DEMO_POP_SEED)


def _pick_customer(customer_id: str | None):
    customers, _hidden = _demo_population()
    if customer_id:
        for c in customers:
            if getattr(c, "customer_id", None) == customer_id:
                return c
        return None
    return customers[0]  # default demo customer


def _build_chat(customer):
    """Build a RetentionChat with a *real* report+diagnosis, driving the same
    sense/diagnose node functions the decision graph and negotiation runner
    use (mirrors `magenta.chat.runner._build_context`) instead of forking a
    second, broken copy of that scoring logic. `sense`/`diagnose` re-load the
    customer via `deps.load_customer(customer_id)`, so this requires that id
    to resolve on the graph's own demo population (`magenta.api.deps`) —
    true here because both populations share DEMO_POP_N/SEED and
    `generate_population` is a seeded, deterministic function (CLAUDE.md:
    "same seed -> identical output"), so the same id always yields an
    equal Customer either way."""
    deps = get_graph_deps()
    state: dict = {"customer_id": customer.customer_id}
    state.update(sense(state, deps))
    state.update(diagnose(state, deps))
    return RetentionChat(deps, customer, state["risk"], state["diagnosis"], authority_cap=80.0)


@router.post("/start", response_model=ChatStartResponse)
def chat_start(
    req: ChatStartRequest,
    tenant: TenantContext = Depends(current_tenant),
) -> ChatStartResponse:
    if req.mode == "persona" and not req.archetype:
        raise HTTPException(422, "archetype required for persona mode")
    customer = _pick_customer(req.customer_id)
    if customer is None:
        raise HTTPException(404, f"unknown customer {req.customer_id}")

    chat_agent = _build_chat(customer)

    persona = None
    if req.mode == "persona":
        try:
            arche = Archetype[req.archetype] if req.archetype in Archetype.__members__ \
                else Archetype(req.archetype)
        except ValueError:
            raise HTTPException(422, f"unknown archetype {req.archetype}")
        _, hidden = _demo_population()
        persona = make_persona(arche, customer, hidden.get(customer.customer_id))

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
