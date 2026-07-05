"""Task 12.5: memory wired into the graph (outcome writes episodic edges;
diagnose reads timeline into its prompt context)."""
from __future__ import annotations

from magenta.graph.nodes import diagnose, outcome
from magenta.graph.state import Diagnosis, RiskUpliftReport, Band, Driver, Timing
from magenta.brain.uplift import Segment
from magenta.offers import Arm


def _outcome_state(deps, holdout: bool) -> dict:
    return {
        "customer_id": "C1", "campaign_id": "K1",
        "offer": deps._offer_fixture, "holdout": holdout,
        "fulfillment": {"status": "FULFILLED"},
    }


def test_outcome_writes_episodic_edges(mem_deps_factory):
    deps = mem_deps_factory()
    state = _outcome_state(deps, holdout=False)
    outcome(state, deps)
    tl = deps.memory.timeline("C1")
    rels = {e.relation for e in tl}
    assert "GAVE" in rels and "OUTCOME" in rels


def test_outcome_holdout_writes_shadow_no_fulfillment_edge(mem_deps_factory):
    """Holdout measures the counterfactual: the oracle never saw an offer, so
    memory must never claim the agent GAVE anything, and the outcome edge is
    tagged holdout_shadow rather than retained/churned (no fulfillment-implying
    content leaks into the customer's memory)."""
    deps = mem_deps_factory()
    state = _outcome_state(deps, holdout=True)
    outcome(state, deps)
    tl = deps.memory.timeline("C1")
    rels = {e.relation for e in tl}
    assert "GAVE" not in rels
    outcome_edges = [e for e in tl if e.relation == "OUTCOME"]
    assert len(outcome_edges) == 1
    assert outcome_edges[0].object == "holdout_shadow"


def _diagnose_state(risk: RiskUpliftReport) -> dict:
    return {
        "customer_id": "C1", "campaign_id": "K1",
        "consent_flags": {"MARKETING": True},
        "risk": risk, "diagnosis": None, "offer": None, "verdict": None,
        "fulfillment": None, "outcome": None, "messages": [], "audit_log": [],
        "requires_approval": False, "holdout": False,
    }


def _risk_report() -> RiskUpliftReport:
    return RiskUpliftReport(
        p_churn=0.72, band=Band.HIGH,
        drivers=[Driver(feature="OVERAGE_EVENTS", label="Overage events",
                        shap_value=0.31, direction="UP")],
        tau_hat=0.18, segment=Segment.PERSUADABLE, engage=True, timing=Timing.ACT_NOW,
    )


HIDDEN_TOKENS = ["theta_churn", "theta_price", "persuadable_segment",
                 "competitor_pull", "θ_churn", "θ_price"]


def test_diagnose_prepends_history_no_hidden_leak(mem_deps_factory):
    deps = mem_deps_factory()
    deps.memory.consolidate("C1", "agent", "GAVE", Arm.BILL_CREDIT.value, "2026-01-01")
    deps.memory.add_edge("C1", "customer", "OUTCOME", "retained", "2026-01-02")

    diagnose(_diagnose_state(_risk_report()), deps)

    joined = " ".join(deps.chat.prompts)
    assert "Prior history" in joined
    assert Arm.BILL_CREDIT.value in joined
    for tok in HIDDEN_TOKENS:
        assert tok.lower() not in joined.lower(), f"hidden token leaked: {tok}"


def test_diagnose_without_memory_is_unaffected(mem_deps_factory):
    """deps.memory is optional (existing callers/tests predate Lab 12) --
    diagnose() must not crash and must not print a history block."""
    deps = mem_deps_factory(memory=None)
    diagnose(_diagnose_state(_risk_report()), deps)
    joined = " ".join(deps.chat.prompts)
    assert "Prior history" not in joined
