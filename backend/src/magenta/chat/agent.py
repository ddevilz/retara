"""RetentionChat — live-negotiation agent (§5.3 + negotiation row).

Persona-free: the prompt built in `_customer_360` is observables (customer
model dump) + `report.drivers` + `diagnosis.narrative` + the current ladder
offer only. It NEVER touches simulator-private hidden state and NEVER quotes
persona brief text (there is no persona-brief field on Customer at all, so
this is structural, not just a discipline).

Safety rule: no fulfillment without CONFIRM_ACT + a *separate, later* explicit
"yes" turn. Reaching CONFIRM_ACT on the same turn a customer signals
acceptance never fulfills by itself — `_awaiting_confirm` is only consumed at
the top of the *next* `respond()` call, so one extra customer turn is always
required before `_fulfill_via_act_node` runs.
"""
from __future__ import annotations

from pydantic import BaseModel

from magenta.llm import chat
from magenta.graph import GraphDeps, RiskUpliftReport, Diagnosis, act
from magenta.sim.population import Customer
from magenta.chat.state import DialogueState, ChatStatus, Turn
from magenta.chat.perceive import perceive
from magenta.chat.controller import next_act, DialogueAct
from magenta.chat.ladder import OfferLadder
from magenta.offers import OfferDecision

_ACCEPT_WORDS = {"yes", "yeah", "sure", "ok", "okay", "do it", "please do", "go ahead", "sounds good"}


class ChatReply(BaseModel):
    text: str
    act: DialogueAct
    offer: OfferDecision | None = None
    state: DialogueState

    model_config = {"arbitrary_types_allowed": True}


def _fulfill_via_act_node(deps: GraphDeps, customer: Customer, offer: OfferDecision) -> dict:
    """Reuse the graph Act node so fulfillment stays idempotent + audited.

    NOTE (brief bug fixed on sight): the brief's state dict omitted
    "campaign_id", but `act()` unconditionally reads `state["campaign_id"]`
    (see magenta.graph.nodes.act) — calling this for real would KeyError.
    Reuse `deps.campaign_id` (the same campaign the rest of the graph uses)
    so idempotency keys line up with any prior graph-driven contact.
    """
    state = {
        "customer_id": customer.customer_id,
        "campaign_id": getattr(deps, "campaign_id", "CHAT"),
        "offer": offer,
        "holdout": False,
        "requires_approval": False,
    }
    return act(state, deps)


def _customer_360(customer: Customer, report: RiskUpliftReport, diagnosis: Diagnosis,
                  offer: OfferDecision | None) -> str:
    obs = customer.model_dump()
    lines = [
        "You are a Deutsche Telekom retention specialist. Persona-free customer 360:",
        f"Observables: {obs}",
        f"Risk drivers: {report.drivers}",
        f"Diagnosis: {diagnosis.narrative}",
    ]
    if offer is not None:
        lines.append(f"Current offer on the table: {offer.arm.value} (EUR {offer.cost:.0f}).")
    lines.append("Be warm, concise, honest. Never invent offers beyond the one shown.")
    return "\n".join(lines)


class RetentionChat:
    def __init__(self, deps: GraphDeps, customer: Customer, report: RiskUpliftReport,
                 diagnosis: Diagnosis, authority_cap: float = 80.0):
        self.deps = deps
        self.customer = customer
        self.report = report
        self.diagnosis = diagnosis
        self.state = DialogueState(customer_id=customer.customer_id)
        self.ladder = OfferLadder(deps.catalog, diagnosis, authority_cap_eur=authority_cap)
        self._current_offer: OfferDecision | None = None
        self._awaiting_confirm = False

    def _wording(self, act_kind: DialogueAct, user_text: str) -> str:
        # param renamed from `act`: shadowed the imported graph `act` node
        # (same naming family as the cli chat-shadow bug).
        system = _customer_360(self.customer, self.report, self.diagnosis, self._current_offer)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Dialogue act: {act_kind.value}. Customer said: {user_text}"},
        ]
        return chat(role="large", messages=messages)

    def respond(self, user_text: str) -> ChatReply:
        self.state.turns.append(Turn(speaker="customer", text=user_text))
        perc = perceive(user_text, self.state)
        self.state.sentiment = perc.sentiment
        self.state.understanding_confidence = perc.understanding_confidence
        for i in perc.intents:
            if i not in self.state.intent_stack:
                self.state.intent_stack.append(i)

        # explicit-yes confirmation gate (safety: irreversible act only after CONFIRM_ACT + yes)
        if self._awaiting_confirm:
            if any(w in user_text.lower() for w in _ACCEPT_WORDS):
                accepted_offer = self._current_offer
                _fulfill_via_act_node(self.deps, self.customer, accepted_offer)
                self.state.status = ChatStatus.ACCEPTED
                text = "Done — I've applied that to your account. Thank you for staying with us."
                return self._emit(text, DialogueAct.CONFIRM_ACT, accepted_offer)
            self._awaiting_confirm = False  # they backed out; keep negotiating

        act_kind = next_act(self.state, perc)

        if act_kind is DialogueAct.HANDOFF:
            self.state.status = ChatStatus.HANDOFF
            return self._emit("Let me bring in a colleague who can help further.", act_kind)

        if act_kind is DialogueAct.NEGOTIATE:
            accepted = "offer_response" in perc.intents and perc.sentiment > 0.4
            # NOTE (brief bug fixed on sight): the brief only checked `accepted`
            # in an `elif` guarded by "there is already a current offer", so a
            # customer who accepts on the very turn the ladder first opens
            # (test_confirm_then_yes_fulfills_once's first turn) fell through
            # to the plain NEGOTIATE reply and the test's 2-turn confirm/yes
            # flow could never reach ACCEPTED. Track `just_opened` instead so
            # "accepted" is checked unconditionally, while concession (which
            # only makes sense against a *pre-existing* offer) stays gated on
            # `not just_opened`.
            just_opened = False
            if self._current_offer is None:
                try:
                    self._current_offer = self.ladder.open()
                except ValueError:
                    # No catalog rung fits the diagnosed causes (e.g. free-form
                    # LLM tags outside the canonical vocab, or genuinely no
                    # fitting offer): escalate gracefully, never crash live chat.
                    self.state.status = ChatStatus.ESCALATED
                    return self._emit(
                        "I want to get this right — let me hand you to a "
                        "specialist with more options.", act_kind)
                just_opened = True

            if accepted:
                self._awaiting_confirm = True
                text = self._wording(DialogueAct.CONFIRM_ACT, user_text)
                self.state.commitments.append(self._current_offer.arm.value)
                return self._emit(text, DialogueAct.CONFIRM_ACT, self._current_offer)
            elif not just_opened:
                nxt = self.ladder.concede(self.state)
                if nxt is None:
                    self.state.status = ChatStatus.ESCALATED
                    return self._emit(
                        "I've reached the limit of what I can offer — let me escalate this "
                        "to a supervisor.", act_kind)
                self._current_offer = nxt
                self.state.concessions_made.append(nxt.arm)
            self.state.ladder_position = self.ladder.position
            text = self._wording(act_kind, user_text)
            return self._emit(text, act_kind, self._current_offer)

        if act_kind in (DialogueAct.EMPATHIZE, DialogueAct.CLARIFY, DialogueAct.ANSWER):
            text = self._wording(act_kind, user_text) if act_kind is DialogueAct.EMPATHIZE else \
                self._deterministic_or_wording(act_kind, user_text)
            return self._emit(text, act_kind, self._current_offer)

        return self._emit("Understood.", act_kind, self._current_offer)

    def _deterministic_or_wording(self, act_kind: DialogueAct, user_text: str) -> str:
        if act_kind is DialogueAct.CLARIFY:
            return "Sorry, I want to make sure I understand — could you say a bit more?"
        return self._wording(act_kind, user_text)

    def _emit(self, text: str, act_kind: DialogueAct, offer: OfferDecision | None = None) -> ChatReply:
        self.state.turns.append(Turn(speaker="agent", text=text))
        return ChatReply(text=text, act=act_kind, offer=offer, state=self.state)
