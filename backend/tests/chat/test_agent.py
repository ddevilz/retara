from unittest.mock import MagicMock, patch

from magenta.brain.risk import Band, Driver
from magenta.brain.uplift import Segment
from magenta.chat.agent import RetentionChat
from magenta.chat.controller import DialogueAct
from magenta.chat.perceive import Perception
from magenta.chat.state import ChatStatus
from magenta.graph import Diagnosis, RiskUpliftReport
from magenta.graph.state import Timing
from magenta.offers import Arm, Offer, OfferCatalog
from magenta.sim.population import generate_population


def _catalog():
    # model_construct (not __new__) — pydantic v2 needs __pydantic_fields_set__
    # etc. initialized; a bare __new__ leaves attribute assignment broken.
    # offers as a list (not the field's declared dict[Arm, Offer]) matches
    # OfferLadder's explicit list-or-dict handling for this exact test shape.
    return OfferCatalog.model_construct(offers=[
        Offer(arm=Arm.ACKNOWLEDGE_AND_FIX, cost=0.0, min_margin=0.0,
              eligibility_note="", fits_causes=["bill_shock"]),
        Offer(arm=Arm.BILL_CREDIT, cost=20.0, min_margin=0.0,
              eligibility_note="", fits_causes=["bill_shock"]),
    ])


def _deps():
    d = MagicMock()
    d.catalog = _catalog()
    return d


def _customer():
    customers, _ = generate_population(1, seed=1)
    return customers[0]


def _report():
    return RiskUpliftReport(
        p_churn=0.7, band=Band.HIGH,
        drivers=[Driver(feature="overage_events_90d", label="Overage events",
                        shap_value=0.3, direction="UP")],
        tau_hat=0.2,
        segment=Segment.PERSUADABLE, engage=True, timing=Timing.ACT_NOW,
    )


def _diag():
    return Diagnosis(
        root_cause_tags=["bill_shock"], narrative="Customer surprised by high bill.",
        eligible_offer_ids=[], confidence=0.8,
    )


def _chat():
    return RetentionChat(_deps(), _customer(), _report(), _diag(), authority_cap=80.0)


def test_negotiate_opens_ladder_with_cheapest_offer():
    perc = Perception(intents=["cancel"], sentiment=0.0, entities={}, understanding_confidence=0.9)
    with patch("magenta.chat.agent.perceive", return_value=perc), \
         patch("magenta.chat.agent.chat", return_value="I hear you.") as m_chat:
        reply = _chat().respond("I'm thinking of cancelling")
    assert reply.act is DialogueAct.NEGOTIATE
    assert reply.offer is not None
    assert reply.offer.arm is Arm.ACKNOWLEDGE_AND_FIX
    # large role used for negotiation wording
    _, kwargs = m_chat.call_args
    role = kwargs.get("role")
    assert role == "large"


def test_no_hidden_or_persona_text_in_agent_prompt():
    perc = Perception(intents=["cancel"], sentiment=0.0, entities={}, understanding_confidence=0.9)
    captured = {}

    def _capture(*a, **k):
        captured["messages"] = k.get("messages")
        return "ok"

    with patch("magenta.chat.agent.perceive", return_value=perc), \
         patch("magenta.chat.agent.chat", side_effect=_capture):
        _chat().respond("cancel please")
    blob = " ".join(m["content"] for m in captured["messages"]).lower()
    for banned in ["theta_churn", "theta_price", "persuadable_segment", "competitor_pull",
                   "accept_threshold", "bluff", "hidden", "brief_text"]:
        assert banned not in blob


def test_ladder_exhausted_escalates_without_forcing_offer():
    # confidence high, keeps negotiating; force concede past cap
    perc = Perception(intents=["offer_response"], sentiment=-0.2, entities={},
                      understanding_confidence=0.9)
    chat = _chat()
    with patch("magenta.chat.agent.perceive", return_value=perc), \
         patch("magenta.chat.agent.chat", return_value="Let me see what I can do."):
        chat.respond("no")   # open ACKNOWLEDGE_AND_FIX
        chat.respond("no")   # concede BILL_CREDIT
        reply = chat.respond("still no")  # next exceeds ladder → escalate
    assert reply.state.status is ChatStatus.ESCALATED


def test_confirm_then_yes_fulfills_once():
    accept = Perception(intents=["offer_response"], sentiment=0.7, entities={},
                        understanding_confidence=0.95)
    chat = _chat()
    with patch("magenta.chat.agent.perceive", return_value=accept), \
         patch("magenta.chat.agent.chat", return_value="Great, shall I apply it?"), \
         patch("magenta.chat.agent._fulfill_via_act_node") as m_fulfill:
        chat.respond("actually yes I like that")   # → CONFIRM_ACT (no fulfill yet)
        assert m_fulfill.call_count == 0
        reply = chat.respond("yes please do it")   # explicit yes → fulfill
    assert m_fulfill.call_count == 1
    assert reply.state.status is ChatStatus.ACCEPTED


def test_irreversible_action_requires_explicit_confirm():
    accept = Perception(intents=["offer_response"], sentiment=0.7, entities={},
                        understanding_confidence=0.95)
    chat = _chat()
    with patch("magenta.chat.agent.perceive", return_value=accept), \
         patch("magenta.chat.agent.chat", return_value="Shall I apply it?"), \
         patch("magenta.chat.agent._fulfill_via_act_node") as m_fulfill:
        chat.respond("sounds good")   # CONFIRM_ACT only
    assert m_fulfill.call_count == 0
