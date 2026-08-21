"""AgentPolicy — the full LLM agent as a drop-in experiment Policy.

Invokes the compiled decision graph for one customer and returns the chosen
OfferDecision (or None when the engage-gate exits early or Guardrail REJECTs).

It deliberately returns only the DECISION. run_experiment owns the 50/50
holdout hash and treatment/holdout accounting against the oracle, so .decide
must not itself fulfill or count outcomes — otherwise we'd double-count.
"""
from __future__ import annotations

from typing import cast

from sqlalchemy.engine import Connection

from magenta.graph.build import GraphDeps, build_graph, persist_audit
from magenta.offers import Arm, OfferDecision
from magenta.sim.population import Customer


class AgentPolicy:
    def __init__(self, deps: GraphDeps):
        self.deps = deps
        self._graph = build_graph(deps)

    def _init_state(self, c: Customer) -> dict:
        return {
            "customer_id": c.customer_id, "campaign_id": self.deps.campaign_id,
            "consent_flags": {"MARKETING": True},
            "risk": None, "diagnosis": None, "offer": None, "verdict": None,
            "fulfillment": None, "outcome": None, "messages": [], "audit_log": [],
            # holdout True here (NOT the harness's RCT holdout flag): it forces
            # the graph's own act/outcome to take the shadow/counterfactual
            # branch so .decide() can never write a real FULFILLMENTS row,
            # record a real contact, or train the bandit a second time. The
            # harness (run_experiment) applies its own treatment/holdout split
            # and oracle accounting downstream of the returned decision.
            "requires_approval": False, "holdout": True,
        }

    def decide(self, c: Customer) -> OfferDecision | None:
        # patch load_customer so the graph resolves THIS customer object.
        def _load_customer(cid: str) -> Customer:
            return c

        self.deps.load_customer = _load_customer
        final = self._graph.invoke(
            self._init_state(c),
            config={"configurable": {"thread_id": f"{self.deps.tenant_id}:{c.customer_id}:{self.deps.campaign_id}"}},
        )
        persist_audit(cast(Connection, self.deps.conn), self.deps.tenant_id, final.get("audit_log", []))
        verdict = final.get("verdict")
        if verdict is not None and verdict.decision == "REJECT":
            return None
        offer = final.get("offer")
        # NO_ACTION must surface as None: run_experiment treats any non-None
        # decision as a real offer (oracle draw + offers_made + acceptance),
        # so leaking NO_ACTION corrupts the Scorecard (review measured 39%
        # of an agent run's "offers" being fake NO_ACTION rows).
        if offer is None or offer.arm is Arm.NO_ACTION:
            return None
        return offer
