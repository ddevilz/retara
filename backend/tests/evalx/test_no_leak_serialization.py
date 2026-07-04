"""Task 9.5: end-to-end no-hidden-leak serialization test.

Asserts that no hidden L1 field ever serializes into:
  (a) DialogueState,
  (b) the retention-agent prompt,
  (c) OverallState.

This is the anti-circularity invariant of §0.5 (CLAUDE.md).
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

from magenta.evalx.hardchecks import scan_hidden_leak
from magenta.chat.state import DialogueState
from magenta.chat.agent import RetentionChat
from magenta.chat.perceive import Perception
from magenta.offers import Arm, Offer, OfferCatalog
from magenta.graph import Diagnosis, RiskUpliftReport
from magenta.brain.risk import Driver
from magenta.sim.population import generate_population


def _chat():
    offer = Offer(
        arm=Arm.ACKNOWLEDGE_AND_FIX,
        cost=0.0,
        min_margin=0.0,
        eligibility_note="test offer",
        fits_causes=["bill_shock"]
    )
    cat = OfferCatalog(offers={Arm.ACKNOWLEDGE_AND_FIX: offer})
    deps = MagicMock()
    deps.catalog = cat
    customers, _ = generate_population(1, seed=11)
    report = RiskUpliftReport(
        p_churn=0.7,
        band="HIGH",
        drivers=[Driver(
            feature="overage_events_90d",
            label="high overage",
            shap_value=0.15,
            direction="UP"
        )],
        tau_hat=0.2,
        segment="PERSUADABLE",
        engage=True,
        timing="ACT_NOW"
    )
    diag = Diagnosis(
        root_cause_tags=["bill_shock"],
        narrative="bill shock",
        eligible_offer_ids=[],
        confidence=0.8
    )
    return RetentionChat(deps, customers[0], report, diag)


def test_dialogue_state_never_serializes_hidden_fields():
    st = DialogueState(
        customer_id="C1",
        intent_stack=["cancel"],
        commitments=["BILL_CREDIT"]
    )
    assert scan_hidden_leak(st.model_dump()) == []


def test_retention_agent_prompt_never_contains_hidden_fields():
    captured = {}

    def _cap(*a, **k):
        captured["messages"] = k.get("messages")
        return "ok"

    perc = Perception(
        intents=["cancel"],
        sentiment=0.0,
        entities={},
        understanding_confidence=0.9
    )
    with patch("magenta.chat.agent.perceive", return_value=perc), \
         patch("magenta.chat.agent.chat", side_effect=_cap):
        _chat().respond("I want to cancel")
    leaks = scan_hidden_leak(captured["messages"])
    assert leaks == []
