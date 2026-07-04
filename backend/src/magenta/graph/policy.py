"""AgentPolicy — the full LLM agent as a drop-in experiment Policy.

Invokes the compiled decision graph for one customer and returns the chosen
OfferDecision (or None when the engage-gate exits early or Guardrail REJECTs).

It deliberately returns only the DECISION. run_experiment owns the 50/50
holdout hash and treatment/holdout accounting against the oracle, so .decide
must not itself fulfill or count outcomes — otherwise we'd double-count.
"""
from __future__ import annotations

from magenta.graph.build import GraphDeps, build_graph, persist_audit
from magenta.offers import OfferDecision


class AgentPolicy:
    def __init__(self, deps: GraphDeps):
        self.deps = deps
        self._graph = build_graph(deps)

    def _init_state(self, c) -> dict:
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

    def decide(self, c) -> OfferDecision | None:
        # patch load_customer so the graph resolves THIS customer object.
        self.deps.load_customer = lambda cid, _c=c: _c
        final = self._graph.invoke(
            self._init_state(c),
            config={"configurable": {"thread_id": f"{c.customer_id}:{self.deps.campaign_id}"}},
        )
        persist_audit(self.deps.conn, final.get("audit_log", []))
        verdict = final.get("verdict")
        if verdict is not None and verdict.decision == "REJECT":
            return None
        return final.get("offer")
