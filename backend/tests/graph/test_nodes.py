import sqlite3

import pytest

from magenta.brain.risk import Band, Driver
from magenta.brain.uplift import Segment
from magenta.graph.nodes import (
    _OBSERVABLE_FIELDS,
    act,
    decide,
    diagnose,
    guardrail,
    idempotency_key,
    outcome,
    sense,
)
from magenta.graph.state import Diagnosis, GuardrailVerdict, RiskUpliftReport, Timing
from magenta.graph.tables import init_graph_tables
from magenta.offers import Arm, OfferDecision
from magenta.sim.population import Customer
import magenta.graph.nodes as nodes_mod


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
    monkeypatch.setattr(nodes_mod, "classify_segment", lambda p, t: Segment.SURE_THING)
    deps = Deps(risk=fakes["risk"], uplift=fakes["uplift"],
                load_customer=lambda cid: customer)
    out = sense(_base_state(customer), deps)
    assert out["risk"].engage is False


def test_diagnose_one_cheap_call(customer, fakes, spy_chat):
    deps = Deps(chat=spy_chat, load_customer=lambda cid: customer)
    state = _base_state(customer)
    # sense would have populated risk; inject it
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


def test_observable_whitelist_matches_customer_model():
    """Every whitelisted prompt field must exist on the REAL Customer model —
    silent None-filtering would gut the diagnose LLM's grounding."""
    for f in _OBSERVABLE_FIELDS:
        assert f in Customer.model_fields, f"{f} not a Customer field"


def test_diagnosis_drops_hallucinated_arms():
    d = Diagnosis(root_cause_tags=["BILL_SHOCK"], narrative="n",
                  eligible_offer_ids=["BILL_CREDIT", "FREE_PONY", "DATA_BOOST"],
                  confidence=0.9)
    assert d.eligible_offer_ids == ["BILL_CREDIT", "DATA_BOOST"]


class Params:
    freq_cap_days = 14
    freq_cap_max = 1
    value_cap = 40.0
    min_margin_floor = 0.0


def _state_with_diagnosis(customer):
    s = _base_state(customer)
    s["risk"] = RiskUpliftReport(
        p_churn=0.72, band=Band.HIGH,
        drivers=[Driver(feature="OVERAGE_EVENTS", label="Overage",
                        shap_value=0.31, direction="UP")],
        tau_hat=0.18, segment=Segment.PERSUADABLE, engage=True, timing=Timing.ACT_NOW)
    s["diagnosis"] = Diagnosis(
        root_cause_tags=["BILL_SHOCK"], narrative="bill shock",
        eligible_offer_ids=[Arm.BILL_CREDIT.value], confidence=0.8)
    return s


def test_decide_intersects_catalog_and_diagnosis(customer, fakes, monkeypatch):
    monkeypatch.setattr(nodes_mod, "featurize", lambda c: [0.0])
    deps = Deps(load_customer=lambda cid: customer, catalog=fakes["catalog"],
                bandit=fakes["bandit"])
    out = decide(_state_with_diagnosis(customer), deps)
    assert isinstance(out["offer"], OfferDecision)
    assert out["offer"].arm is Arm.BILL_CREDIT      # only common arm
    assert out["audit_log"][0]["NODE"] == "DECIDE"


def test_guardrail_passes_clean(customer, fakes):
    deps = Deps(load_customer=lambda cid: customer, catalog=fakes["catalog"],
                conn=_mem_conn(), params=Params())
    s = _state_with_diagnosis(customer)
    s["offer"] = OfferDecision(arm=Arm.BILL_CREDIT, cost=8.0, propensity=0.6)
    out = guardrail(s, deps)
    assert out["verdict"].decision == "PASS"


def test_guardrail_rejects_on_margin(customer, fakes):
    cat = fakes["catalog"]; cat._min_margin = 5.0
    # margin 22 - cost 20 = 2 < 5 => reject
    deps = Deps(load_customer=lambda cid: customer, catalog=cat,
                conn=_mem_conn(), params=Params())
    s = _state_with_diagnosis(customer)
    s["offer"] = OfferDecision(arm=Arm.BILL_CREDIT, cost=20.0, propensity=0.6)
    out = guardrail(s, deps)
    assert out["verdict"].decision == "REJECT"
    assert "MIN_MARGIN" in out["verdict"].failed_policies


def test_guardrail_rejects_on_consent(customer, fakes):
    deps = Deps(load_customer=lambda cid: customer, catalog=fakes["catalog"],
                conn=_mem_conn(), params=Params())
    s = _state_with_diagnosis(customer)
    s["consent_flags"] = {"MARKETING": False}
    s["offer"] = OfferDecision(arm=Arm.BILL_CREDIT, cost=8.0, propensity=0.6)
    out = guardrail(s, deps)
    assert out["verdict"].decision == "REJECT"
    assert "CONSENT" in out["verdict"].failed_policies


def test_guardrail_value_cap_needs_approval(customer, fakes):
    deps = Deps(load_customer=lambda cid: customer, catalog=fakes["catalog"],
                conn=_mem_conn(), params=Params())
    s = _state_with_diagnosis(customer)
    s["offer"] = OfferDecision(arm=Arm.DEVICE_UPGRADE, cost=50.0, propensity=0.6)
    # cost 50 > value_cap 40 => NEEDS_APPROVAL (but margin: 22-50 <0 also fails)
    # so give a big-margin customer:
    # NOTE: brief snippet set customer.gross_margin (not a real Customer field,
    # would be a silent no-op combined with the getattr(..., "gross_margin", 0.0)
    # bug in guardrail()). The real field is gross_margin_monthly.
    customer.gross_margin_monthly = 100.0
    out = guardrail(s, deps)
    assert out["verdict"].decision == "NEEDS_APPROVAL"
    assert out["requires_approval"] is True


def _mem_conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    init_graph_tables(c)
    return c


## --------------------------------------------------------------------------- #
## ACT — idempotent fulfillment; holdout => shadow-log only.
## --------------------------------------------------------------------------- #
def test_idempotency_key_stable():
    k1 = idempotency_key("CUST-1", "CAMP-A", Arm.BILL_CREDIT)
    k2 = idempotency_key("CUST-1", "CAMP-A", Arm.BILL_CREDIT)
    k3 = idempotency_key("CUST-1", "CAMP-A", Arm.DATA_BOOST)
    assert k1 == k2 and k1 != k3 and len(k1) == 64


def test_act_fulfills_once_on_double_invoke(customer, fakes):
    conn = _mem_conn()
    deps = Deps(load_customer=lambda cid: customer, catalog=fakes["catalog"], conn=conn)
    s = _state_with_diagnosis(customer)
    s["offer"] = OfferDecision(arm=Arm.BILL_CREDIT, cost=8.0, propensity=0.6)
    s["verdict"] = GuardrailVerdict(decision="PASS")
    o1 = act(s, deps)
    o2 = act(s, deps)   # re-run (simulates interrupt/resume)
    n = conn.execute("SELECT count(*) FROM FULFILLMENTS").fetchone()[0]
    assert n == 1
    assert o1["fulfillment"]["IDEMPOTENCY_KEY"] == o2["fulfillment"]["IDEMPOTENCY_KEY"]


def test_act_holdout_shadow_no_row_but_audit(customer, fakes):
    conn = _mem_conn()
    deps = Deps(load_customer=lambda cid: customer, catalog=fakes["catalog"], conn=conn)
    s = _state_with_diagnosis(customer)
    s["offer"] = OfferDecision(arm=Arm.BILL_CREDIT, cost=8.0, propensity=0.6)
    s["verdict"] = GuardrailVerdict(decision="PASS")
    s["holdout"] = True
    out = act(s, deps)
    n = conn.execute("SELECT count(*) FROM FULFILLMENTS").fetchone()[0]
    assert n == 0
    assert out["fulfillment"]["status"] == "SHADOW"
    assert out["audit_log"][0]["NODE"] == "ACT"


## --------------------------------------------------------------------------- #
## OUTCOME — oracle result -> reward -> bandit.update -> audit.
## --------------------------------------------------------------------------- #
def test_outcome_updates_bandit_and_computes_reward(customer, fakes, monkeypatch):
    # NOTE: brief snippet imported nodes_mod/types inside the test body but never
    # used them to stub featurize() — FakeCustomer lacks the raw fields featurize()
    # reads directly (total_charges, data_gb_used_p50, ...), so calling the real
    # featurize() here would AttributeError. Stub it, same as test_decide_* does.
    monkeypatch.setattr(nodes_mod, "featurize", lambda c: [0.0])
    deps = Deps(load_customer=lambda cid: customer, oracle=fakes["oracle"],
                bandit=fakes["bandit"])
    s = _state_with_diagnosis(customer)
    s["offer"] = OfferDecision(arm=Arm.BILL_CREDIT, cost=8.0, propensity=0.6)
    out = outcome(s, deps)
    # accepted=True, churned=False => retained; reward = margin(22)*12 - cost(8) = 256
    # (annualized to match the Lab-5 bandit-episodes reward scale)
    assert out["outcome"]["reward"] == pytest.approx(256.0)
    assert out["outcome"]["retained"] is True
    assert fakes["bandit"].updates == [(Arm.BILL_CREDIT, pytest.approx(256.0))]


def test_outcome_holdout_no_bandit_update(customer, fakes):
    deps = Deps(load_customer=lambda cid: customer, oracle=fakes["oracle"],
                bandit=fakes["bandit"])
    s = _state_with_diagnosis(customer)
    s["offer"] = OfferDecision(arm=Arm.BILL_CREDIT, cost=8.0, propensity=0.6)
    s["holdout"] = True
    outcome(s, deps)
    assert fakes["bandit"].updates == []   # holdout never trains the bandit
