import sqlite3

import pytest

from magenta.brain.risk import Band, Driver
from magenta.brain.uplift import Segment
from magenta.cost.cache import SemanticCache
from magenta.graph.batch_diagnose import diagnose_cohort, driver_signature
from magenta.graph.state import RiskUpliftReport, Timing
from magenta.memory.embed import LocalEmbedder


def _report(feat="OVERAGE_EVENTS", shap=0.3, band=Band.HIGH, extra_driver=None):
    drivers = [Driver(feature=feat, label=feat, shap_value=shap, direction="UP")]
    if extra_driver is not None:
        drivers.append(extra_driver)
    return RiskUpliftReport(
        p_churn=0.7, band=band, drivers=drivers,
        tau_hat=0.18, segment=Segment.PERSUADABLE, engage=True, timing=Timing.ACT_NOW)


class C:
    def __init__(self, cid):
        self.customer_id = cid


@pytest.fixture(scope="module")
def _embedder():
    return LocalEmbedder()


def _cache(embedder, threshold: float = 0.75):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return SemanticCache(conn, embedder, threshold=threshold)


def test_signature_stable_and_shape_sensitive():
    a = driver_signature(_report(shap=0.30))
    b = driver_signature(_report(shap=0.90))     # same sign+dir => same signature
    c = driver_signature(_report(shap=-0.30))    # sign flip => different
    assert a == b
    assert a != c


def test_cache_hit_one_call_for_identical_signatures(monkeypatch, _embedder):
    """Task 13.4: byte-identical driver shape -> SemanticCache exact match
    (cosine 1.0) -> only the first customer pays for a real LLM call."""
    calls = []
    monkeypatch.setattr("magenta.graph.batch_diagnose._chat",
                        lambda role, msgs: (calls.append(role), "BILL_SHOCK")[1])
    customers = [C("CUST-1"), C("CUST-2")]
    reports = {"CUST-1": _report(shap=0.3), "CUST-2": _report(shap=0.7)}  # same sig
    out = diagnose_cohort(customers, reports, deps=object(), cache=_cache(_embedder))
    assert set(out) == {"CUST-1", "CUST-2"}
    assert len(calls) == 1     # semantic cache: identical driver shape => 1 LLM call


def test_near_duplicate_driver_shape_also_collapses(monkeypatch, _embedder):
    """The semantic upgrade over an exact-hash cache: a customer whose report
    adds one minor secondary driver (different driver_signature, but a
    near-duplicate diagnosis text) still reuses the cached answer."""
    calls = []
    monkeypatch.setattr("magenta.graph.batch_diagnose._chat",
                        lambda role, msgs: (calls.append(role), "BILL_SHOCK")[1])
    extra = Driver(feature="DROPPED_CALLS", label="DROPPED_CALLS", shap_value=0.02, direction="UP")
    customers = [C("CUST-1"), C("CUST-2")]
    reports = {"CUST-1": _report(), "CUST-2": _report(extra_driver=extra)}
    assert driver_signature(reports["CUST-1"]) != driver_signature(reports["CUST-2"])
    diagnose_cohort(customers, reports, deps=object(), cache=_cache(_embedder))
    assert len(calls) == 1


def test_without_cache_each_customer_gets_an_independent_call(monkeypatch):
    """Caching is opt-in (pass `cache=`): with none given, diagnose_cohort no
    longer performs any in-process exact-signature dedup on its own -- that
    responsibility moved entirely to the SemanticCache in Task 13.4."""
    calls = []
    monkeypatch.setattr("magenta.graph.batch_diagnose._chat",
                        lambda role, msgs: (calls.append(role), "BILL_SHOCK")[1])
    customers = [C("CUST-1"), C("CUST-2")]
    reports = {"CUST-1": _report(shap=0.3), "CUST-2": _report(shap=0.3)}  # identical sig, no cache
    diagnose_cohort(customers, reports, deps=object(), cache=None)
    assert len(calls) == 2
