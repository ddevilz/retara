import pytest

from magenta.brain.risk import Band, Driver
from magenta.brain.uplift import Segment
from magenta.graph.batch_diagnose import diagnose_cohort, driver_signature
from magenta.graph.state import Diagnosis, RiskUpliftReport, Timing


def _report(feat="OVERAGE_EVENTS", shap=0.3, band=Band.HIGH):
    return RiskUpliftReport(
        p_churn=0.7, band=band,
        drivers=[Driver(feature=feat, label=feat, shap_value=shap, direction="UP")],
        tau_hat=0.18, segment=Segment.PERSUADABLE, engage=True, timing=Timing.ACT_NOW)


class CountingChat:
    def __init__(self):
        self.n = 0

    def chat_structured(self, role, messages, model_cls):
        self.n += 1
        return Diagnosis(root_cause_tags=["BILL_SHOCK"], narrative="n",
                         eligible_offer_ids=["BILL_CREDIT"], confidence=0.8)


class C:
    def __init__(self, cid):
        self.customer_id = cid


def test_signature_stable_and_shape_sensitive():
    a = driver_signature(_report(shap=0.30))
    b = driver_signature(_report(shap=0.90))     # same sign+dir => same signature
    c = driver_signature(_report(shap=-0.30))    # sign flip => different
    assert a == b
    assert a != c


def test_cache_hit_one_call_for_identical_signatures():
    chat = CountingChat()
    customers = [C("CUST-1"), C("CUST-2")]
    reports = {"CUST-1": _report(shap=0.3), "CUST-2": _report(shap=0.7)}  # same sig
    out = diagnose_cohort(customers, reports, chat, max_workers=2)
    assert set(out) == {"CUST-1", "CUST-2"}
    assert chat.n == 1     # cache: 2 identical signatures => 1 LLM call


def test_distinct_signatures_two_calls():
    chat = CountingChat()
    customers = [C("CUST-1"), C("CUST-2")]
    reports = {"CUST-1": _report(shap=0.3), "CUST-2": _report(shap=-0.3)}  # diff sig
    diagnose_cohort(customers, reports, chat, max_workers=2)
    assert chat.n == 2
