"""Ablation-ladder policies (§7 credibility centerpiece).

Ladder: no-action -> rules -> risk+rules -> agent (System-2 off) -> agent (System-2
on). Each rung must EARN its complexity; if the agent doesn't beat rules we
report that honestly (council amendment #1 fixes this exact 5-rung order).

`agent_s1` and `agent` are both `AgentPolicy` — the only difference is the
`system2_enabled` flag on `GraphDeps`, so `agent_s1` measures the full graph
(risk/uplift/bandit/guardrail/LLM-diagnose) on its own, and `agent` measures
System-2's marginal contribution on top of that once Task 7.6 wires
`should_deliberate`/`deliberate` into the `decide` node. Until then `agent`
runs the same graph as `agent_s1` with the flag simply set True (a no-op
until 7.6 reads it).
"""
from __future__ import annotations

from dataclasses import replace

from magenta.graph.build import GraphDeps
from magenta.graph.policy import AgentPolicy
from magenta.offers import Arm, OfferDecision

RUNGS = ["noaction", "rules", "risk_rules", "agent_s1", "agent"]


def _fixed_credit() -> OfferDecision:
    return OfferDecision(arm=Arm.BILL_CREDIT, cost=8.0,
                          rationale="fixed bill credit", propensity=1.0)


class NoActionPolicy:
    """Ladder floor: never contacts anyone."""

    def decide(self, c) -> OfferDecision | None:
        return None


class RulesPolicy:
    """Naive ladder rung: fire a fixed credit on obvious signals (contract-end
    or overage) with no risk model at all.

    Reads the real Customer's `overage_events_90d`; falls back to a plain
    `overage_events` attribute so this also works against the minimal
    duck-typed customer stubs used in tests.
    """

    def decide(self, c) -> OfferDecision | None:
        end = getattr(c, "contract_end_days", 999)
        overage = getattr(c, "overage_events_90d", None)
        if overage is None:
            overage = getattr(c, "overage_events", 0)
        if end < 30 or overage > 0:
            return _fixed_credit()
        return None


class RiskRulesPolicy:
    """Risk-gate + fixed BILL_CREDIT: act only when p_churn >= threshold."""

    def __init__(self, risk, threshold: float = 0.5):
        self.risk = risk
        self.threshold = threshold

    def decide(self, c) -> OfferDecision | None:
        if self.risk.score(c).p_churn >= self.threshold:
            return _fixed_credit()
        return None


def make_policy(rung: str, deps: GraphDeps):
    """Ladder-rung factory (§7).

    `deps` must be a real `GraphDeps` for `risk_rules`/`agent_s1`/`agent`
    (risk_rules reads `deps.risk`; the agent rungs `dataclasses.replace` it to
    flip `system2_enabled`). `noaction`/`rules` never touch `deps`.
    """
    if rung == "noaction":
        return NoActionPolicy()
    if rung == "rules":
        return RulesPolicy()
    if rung == "risk_rules":
        return RiskRulesPolicy(deps.risk)
    if rung == "agent_s1":
        return AgentPolicy(replace(deps, system2_enabled=False))
    if rung == "agent":
        return AgentPolicy(replace(deps, system2_enabled=True))
    raise ValueError(f"unknown rung: {rung}")
