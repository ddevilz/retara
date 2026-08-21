from unittest.mock import MagicMock, patch

from magenta.brain.risk import Band, Driver
from magenta.chat.agent import RetentionChat
from magenta.chat.perceive import Perception
from magenta.chat.persona import Archetype, PersonaAgent, make_persona
from magenta.chat.runner import NegotiationResult, run_negotiation
from magenta.chat.state import ChatStatus
from magenta.graph import Diagnosis, RiskUpliftReport
from magenta.graph.state import Timing
from magenta.offers import Arm, Offer, OfferCatalog
from magenta.sim.population import HiddenState, Segment, generate_population


def _catalog():
    # NOTE (brief bug fixed on sight, same fix already applied in
    # tests/chat/test_agent.py): `OfferCatalog.__new__` + a hand-set
    # `.offers` attribute skips pydantic v2's __init__, leaving
    # __pydantic_fields_set__ uninitialized -> AttributeError on the first
    # attribute access. Use `model_construct` instead. Also `Offer(...)`
    # requires `eligibility_note` (no default) -- the brief's call omitted
    # it, which fails required-field validation.
    return OfferCatalog.model_construct(offers=[
        Offer(arm=Arm.ACKNOWLEDGE_AND_FIX, cost=0.0, min_margin=0.0,
              eligibility_note="", fits_causes=["bill_shock"]),
    ])


def _deps():
    d = MagicMock()
    d.catalog = _catalog()
    return d


def _fixtures():
    customers, _ = generate_population(1, seed=5)
    c = customers[0]
    # NOTE (brief bug fixed on sight): `drivers=["x"]` doesn't validate --
    # RiskUpliftReport.drivers is `list[Driver]`, a pydantic model, not
    # list[str]. Use a real Driver instance.
    report = RiskUpliftReport(p_churn=0.7, band="HIGH",
                              drivers=[Driver(feature="x", label="x", shap_value=0.1,
                                             direction="UP")],
                              tau_hat=0.2, segment="PERSUADABLE", engage=True,
                              timing="ACT_NOW")
    diag = Diagnosis(root_cause_tags=["bill_shock"], narrative="bill shock",
                     eligible_offer_ids=[], confidence=0.8)
    return c, report, diag


def test_runner_terminates_within_max_turns():
    c, report, diag = _fixtures()
    hidden = HiddenState(theta_churn_base=0.4, theta_price_sens=0.9,
                         persuadable_segment=Segment.PERSUADABLE, competitor_pull=0.2)
    persona = PersonaAgent(make_persona(Archetype.PRICE_HAGGLER, c, hidden))
    with patch("magenta.chat.runner._build_context", return_value=(c, report, diag)), \
         patch("magenta.chat.persona.chat", return_value="No, too expensive."), \
         patch("magenta.chat.agent.chat", return_value="Here is what I can do."), \
         patch("magenta.chat.agent.perceive") as m_perc:
        from magenta.chat.perceive import Perception
        m_perc.return_value = Perception(intents=["cancel"], sentiment=-0.1, entities={},
                                         understanding_confidence=0.9)
        result = run_negotiation(_deps(), c, persona, max_turns=6)
    assert isinstance(result, NegotiationResult)
    assert result.turns_used <= 6
    assert result.status in set(ChatStatus)


def test_sleeping_dog_not_forced_an_offer():
    c, report, diag = _fixtures()
    hidden = HiddenState(theta_churn_base=0.4, theta_price_sens=0.5,
                         persuadable_segment=Segment.SLEEPING_DOG, competitor_pull=0.0)
    persona = PersonaAgent(make_persona(Archetype.SLEEPING_DOG, c, hidden))
    with patch("magenta.chat.runner._build_context", return_value=(c, report, diag)), \
         patch("magenta.chat.persona.chat", return_value="I'm fine, please leave me alone."), \
         patch("magenta.chat.agent.chat", return_value="Understood, sorry to bother you."), \
         patch("magenta.chat.agent.perceive") as m_perc:
        from magenta.chat.perceive import Perception
        m_perc.return_value = Perception(intents=["chitchat"], sentiment=-0.2, entities={},
                                         understanding_confidence=0.9)
        result = run_negotiation(_deps(), c, persona, max_turns=6)
    assert result.status in (ChatStatus.REJECTED, ChatStatus.HANDOFF)
    assert result.offer_final is None


def test_chat_shim_binds_llm_chat_not_cli_command():
    """Regression: the `magenta chat` CLI command once shadowed magenta.llm.chat
    at module level, so _ChatShim routed LLM calls into the typer command and
    crashed live cohort runs (string seed into generate_population)."""
    import magenta.cli as cli_mod
    import magenta.llm as llm_mod

    assert cli_mod.chat is llm_mod.chat, "llm.chat is shadowed in magenta.cli"




def _chat_bot(tags=None):
    catalog = OfferCatalog.model_construct(offers=[
        Offer(arm=Arm.ACKNOWLEDGE_AND_FIX, cost=0.0, min_margin=0.0,
              eligibility_note="", fits_causes=["BILL_SHOCK"]),
        Offer(arm=Arm.BILL_CREDIT, cost=20.0, min_margin=0.0,
              eligibility_note="", fits_causes=["BILL_SHOCK"]),
    ])
    deps = MagicMock()
    deps.catalog = catalog
    customers, _ = generate_population(1, seed=1)
    report = RiskUpliftReport(
        p_churn=0.7, band=Band.HIGH,
        drivers=[Driver(feature="OVERAGE_EVENTS_90D", label="Overage events",
                        shap_value=0.3, direction="UP")],
        tau_hat=0.2, segment=Segment.PERSUADABLE, engage=True, timing=Timing.ACT_NOW)
    diag = Diagnosis(root_cause_tags=tags or ["BILL_SHOCK"], narrative="n",
                     eligible_offer_ids=[], confidence=0.8)
    return RetentionChat(deps, customers[0], report, diag, authority_cap=80.0)


def _negotiate_perc():
    return Perception(intents=["cancel", "offer_response"], sentiment=0.6,
                      entities={}, understanding_confidence=0.9)


def test_offer_final_populated_on_accept():
    """Regression: ACCEPTED negotiations must carry the accepted offer out
    (ChatReply.offer was always None on the accepting turn — lab 8+9 gate)."""
    bot = _chat_bot()
    with patch("magenta.chat.agent.perceive", return_value=_negotiate_perc()),          patch("magenta.chat.agent.chat", return_value="offer text"),          patch("magenta.chat.agent._fulfill_via_act_node"):
        bot.respond("that bill is too high, what can you do?")   # opens + CONFIRM_ACT
        reply = bot.respond("yes please")                         # explicit yes
    assert bot.state.status is ChatStatus.ACCEPTED
    assert reply.offer is not None and reply.offer.cost >= 0.0


def test_ladder_no_fit_escalates_not_crashes():
    """Regression: diagnosis tags with zero fitting rungs must escalate
    gracefully, not crash live chat (lab 8+9 gate)."""
    bot = _chat_bot(tags=["SOMETHING_UNMAPPED"])
    with patch("magenta.chat.agent.perceive", return_value=_negotiate_perc()),          patch("magenta.chat.agent.chat", return_value="hmm"):
        reply = bot.respond("I want to cancel unless you give me a deal")
    assert bot.state.status is ChatStatus.ESCALATED
    assert reply.offer is None
