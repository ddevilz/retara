from unittest.mock import patch, MagicMock

from magenta.chat.runner import run_negotiation, NegotiationResult
from magenta.chat.state import ChatStatus
from magenta.chat.persona import PersonaAgent, make_persona, Archetype
from magenta.offers import Arm, Offer, OfferCatalog
from magenta.graph import Diagnosis, RiskUpliftReport
from magenta.brain.risk import Driver
from magenta.sim.population import generate_population, HiddenState, Segment


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
