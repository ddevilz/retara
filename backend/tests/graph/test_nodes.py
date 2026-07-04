import pytest

from magenta.brain.uplift import Segment
from magenta.graph.nodes import diagnose, sense
from magenta.graph.state import Diagnosis
from magenta.offers import Arm


## --- tiny deps holder used only for node unit tests -------------------------
class Deps:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _base_state(customer, **over):
    s = {
        "customer_id": customer.customer_id,
        "campaign_id": "CAMP-A",
        "consent_flags": {"MARKETING": True},
        "risk": None, "diagnosis": None, "offer": None, "verdict": None,
        "fulfillment": None, "outcome": None, "messages": [], "audit_log": [],
        "requires_approval": False, "holdout": False,
    }
    s.update(over)
    return s


def test_sense_engages_persuadable(customer, fakes):
    deps = Deps(risk=fakes["risk"], uplift=fakes["uplift"],
                load_customer=lambda cid: customer)
    out = sense(_base_state(customer), deps)
    r = out["risk"]
    assert r.engage is True
    assert r.segment is Segment.PERSUADABLE
    assert len(out["audit_log"]) == 1
    assert out["audit_log"][0]["NODE"] == "SENSE"


def test_sense_does_not_engage_sure_thing(customer, fakes, monkeypatch):
    # force classify_segment -> SURE_THING regardless of numbers
    import magenta.graph.nodes as nodes_mod
    monkeypatch.setattr(nodes_mod, "classify_segment", lambda p, t: Segment.SURE_THING)
    deps = Deps(risk=fakes["risk"], uplift=fakes["uplift"],
                load_customer=lambda cid: customer)
    out = sense(_base_state(customer), deps)
    assert out["risk"].engage is False


def test_diagnose_one_cheap_call(customer, fakes, spy_chat):
    deps = Deps(chat=spy_chat, load_customer=lambda cid: customer)
    state = _base_state(customer)
    # sense would have populated risk; inject it
    from magenta.graph.state import RiskUpliftReport, Timing
    from magenta.brain.risk import Band, Driver
    state["risk"] = RiskUpliftReport(
        p_churn=0.72, band=Band.HIGH,
        drivers=[Driver(feature="OVERAGE_EVENTS", label="Overage events",
                        shap_value=0.31, direction="UP")],
        tau_hat=0.18, segment=Segment.PERSUADABLE, engage=True, timing=Timing.ACT_NOW,
    )
    out = diagnose(state, deps)
    assert isinstance(out["diagnosis"], Diagnosis)
    assert len(spy_chat.calls) == 1           # exactly ONE cheap call
    assert spy_chat.calls[0]["role"] == "cheap"
    assert out["audit_log"][0]["NODE"] == "DIAGNOSE"


HIDDEN_TOKENS = ["theta_churn", "theta_price", "persuadable_segment",
                 "competitor_pull", "θ_churn", "θ_price"]


def test_diagnose_prompt_has_no_hidden_leak(customer, fakes, spy_chat):
    deps = Deps(chat=spy_chat, load_customer=lambda cid: customer)
    state = _base_state(customer)
    from magenta.graph.state import RiskUpliftReport, Timing
    from magenta.brain.risk import Band, Driver
    state["risk"] = RiskUpliftReport(
        p_churn=0.72, band=Band.HIGH,
        drivers=[Driver(feature="OVERAGE_EVENTS", label="Overage events",
                        shap_value=0.31, direction="UP")],
        tau_hat=0.18, segment=Segment.PERSUADABLE, engage=True, timing=Timing.ACT_NOW,
    )
    diagnose(state, deps)
    joined = " ".join(spy_chat.prompts).lower()
    for tok in HIDDEN_TOKENS:
        assert tok.lower() not in joined, f"hidden token leaked: {tok}"
