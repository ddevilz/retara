from unittest.mock import patch

from magenta.chat.persona import Archetype, PersonaAgent, PersonaBrief, make_persona
from magenta.chat.state import DialogueState
from magenta.sim.population import HiddenState, Segment, generate_population


def _cust_hidden():
    customers, store = generate_population(1, seed=3)
    c = customers[0]
    hidden = store.get(c.customer_id) if hasattr(store, "get") else HiddenState(
        theta_churn_base=0.4, theta_price_sens=0.8,
        persuadable_segment=Segment.PERSUADABLE, competitor_pull=0.3,
    )
    return c, hidden


def test_archetype_enum_values():
    assert {a.value for a in Archetype} == {
        "BILL_SHOCK", "CONFUSED", "PRICE_HAGGLER", "NETWORK_COMPLAINER",
        "COMPETITOR_BLUFFER", "SLEEPING_DOG",
    }


def test_make_persona_builds_brief_from_hidden():
    c, hidden = _cust_hidden()
    brief = make_persona(Archetype.BILL_SHOCK, c, hidden)
    assert isinstance(brief, PersonaBrief)
    assert brief.archetype is Archetype.BILL_SHOCK
    assert 0.0 <= brief.price_sensitivity <= 1.0
    assert brief.accept_threshold_eur >= 0.0
    assert brief.brief_text


def test_persona_reply_uses_large_role_and_hidden_system_prompt():
    c, hidden = _cust_hidden()
    brief = make_persona(Archetype.COMPETITOR_BLUFFER, c, hidden)
    with patch("magenta.chat.persona.chat", return_value="Vodafone offered me cheaper.") as m:
        out = PersonaAgent(brief).reply("Can I help you stay?", DialogueState(customer_id=c.customer_id))
    assert out == "Vodafone offered me cheaper."
    _, kwargs = m.call_args
    assert kwargs.get("role") == "large"
    sys_prompt = kwargs["messages"][0]["content"]
    assert "NEVER" in sys_prompt.upper()
    assert brief.true_cause in sys_prompt
