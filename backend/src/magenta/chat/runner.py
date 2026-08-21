"""Negotiation runner — drives RetentionChat against either a synthetic
PersonaAgent (offline eval / scripted demo) or a live human via stdin
(§5.3 chat loop, Lab 8 exit gate).

`persona=None` selects HUMAN mode: the human types replies on stdin and the
agent's lines print to stdout as they're produced. Any non-None
`PersonaAgent` selects scripted-adversary mode: the loop alternates
persona -> agent turns silently; callers (tests, `magenta chat`) render the
returned `NegotiationResult.transcript` however they like.
"""
from __future__ import annotations

from typing import cast

from pydantic import BaseModel

from magenta.chat.agent import RetentionChat
from magenta.chat.persona import PersonaAgent
from magenta.chat.state import ChatStatus, Turn
from magenta.graph import Diagnosis, RiskUpliftReport, diagnose, sense
from magenta.graph.state import OverallState
from magenta.offers import OfferDecision
from magenta.sim.population import Customer


class NegotiationResult(BaseModel):
    status: ChatStatus
    transcript: list[Turn]
    offer_final: OfferDecision | None = None
    turns_used: int = 0

    model_config = {"arbitrary_types_allowed": True}


_OPENER = ("Hi, this is Deutsche Telekom. We noticed something on your account and "
           "wanted to check in — is now a good time?")


def _build_context(deps, customer: Customer) -> tuple[Customer, RiskUpliftReport, Diagnosis]:
    """Score + diagnose the customer for the chat, by driving the same
    sense/diagnose node functions the full decision graph uses (§5.5), so
    negotiation scoring never re-derives its own copy of that logic. Kept as
    a top-level function (not inlined into `run_negotiation`) so tests can
    stub it wholesale.

    NOTE: `sense`/`diagnose` re-load the customer from `deps.load_customer`
    by id rather than taking a Customer directly, so this requires
    `deps.load_customer(customer.customer_id)` to resolve back to
    `customer` -- the same wiring `magenta run-one`/`magenta chat` already
    use.
    """
    # Deliberately partial: sense()/diagnose() are driven incrementally here the
    # same way build_graph's compiled StateGraph drives them turn-by-turn --
    # each call only reads the keys the prior call already populated.
    state: dict = {"customer_id": customer.customer_id}
    state.update(sense(cast(OverallState, state), deps))
    state.update(diagnose(cast(OverallState, state), deps))
    return customer, state["risk"], state["diagnosis"]


def run_negotiation(deps, customer: Customer, persona: PersonaAgent | None,
                     max_turns: int = 8) -> NegotiationResult:
    """Alternate agent<->persona (or agent<->human) turns until the dialogue
    resolves (ACCEPTED/REJECTED/ESCALATED/HANDOFF) or `max_turns` turns are
    spent, whichever comes first. Always opens with a proactive agent line.
    """
    cust, report, diag = _build_context(deps, customer)
    chat_agent = RetentionChat(deps, cust, report, diag)

    agent_text = _OPENER
    final_offer: OfferDecision | None = None
    turns_used = 0

    if persona is None:
        # NOTE (brief bug fixed on sight): the brief's human-mode loop went
        # straight to `input("you> ")` without ever printing `_OPENER` first,
        # so a live operator would be asked to reply with no visible context
        # of what the agent supposedly just said -- contradicts the loop
        # contract ("open with a proactive agent line"). Print it up front.
        print(f"agent> {agent_text}")

    for _ in range(max_turns):
        if persona is None:
            user_text = input("you> ").strip()
            if not user_text:
                break
        else:
            user_text = persona.reply(agent_text, chat_agent.state)

        reply = chat_agent.respond(user_text)
        turns_used += 1
        agent_text = reply.text
        if reply.state.status is ChatStatus.ACCEPTED and reply.offer is not None:
            final_offer = reply.offer
        if persona is None:
            print(f"agent> {agent_text}")
        if reply.state.status is not ChatStatus.ACTIVE:
            break

    status = chat_agent.state.status
    if status is ChatStatus.ACTIVE:  # ran out of turns without resolution
        status = ChatStatus.HANDOFF
    if status is not ChatStatus.ACCEPTED:
        final_offer = None

    return NegotiationResult(
        status=status,
        transcript=list(chat_agent.state.turns),
        offer_final=final_offer,
        turns_used=turns_used,
    )
