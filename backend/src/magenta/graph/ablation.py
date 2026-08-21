"""Ablation-ladder policies (§7 credibility centerpiece).

Ladder: no-action -> rules -> risk+rules -> agent (System-2 off) -> agent (System-2
on). Each rung must EARN its complexity; if the agent doesn't beat rules we
report that honestly (council amendment #1 fixes this exact 5-rung order).

`agent_s1` and `agent` are both `AgentPolicy` — the only difference is the
`system2_enabled` flag on `GraphDeps`, so `agent_s1` measures the full graph
(risk/uplift/bandit/guardrail/LLM-diagnose) on its own, and `agent` measures
System-2's marginal contribution on top of that: the `decide` node (Task 7.6,
`magenta.graph.nodes.decide`) routes to `system2.deliberate` whenever
`system2_enabled` is True AND `system2.should_deliberate(...)` triggers
(high CLV >= `deps.params.p90_clv` or low diagnosis confidence) — otherwise
both rungs fall through to the identical bandit path.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace

from magenta.experiment import Scorecard, run_experiment
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
        end: int = getattr(c, "contract_end_days", 999)
        overage_90d = getattr(c, "overage_events_90d", None)
        overage: int = overage_90d if overage_90d is not None else getattr(c, "overage_events", 0)
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


def make_policy(rung: str, deps: GraphDeps | None):
    """Ladder-rung factory (§7).

    `deps` must be a real `GraphDeps` for `risk_rules`/`agent_s1`/`agent`
    (risk_rules reads `deps.risk`; the agent rungs `dataclasses.replace` it to
    flip `system2_enabled`). `noaction`/`rules` never touch `deps`.
    """
    if rung == "noaction":
        return NoActionPolicy()
    if rung == "rules":
        return RulesPolicy()
    if rung not in ("risk_rules", "agent_s1", "agent"):
        raise ValueError(f"unknown rung: {rung}")
    if deps is None:
        raise ValueError(f"rung {rung!r} requires a real GraphDeps")
    if rung == "risk_rules":
        return RiskRulesPolicy(deps.risk)
    if rung == "agent_s1":
        return AgentPolicy(replace(deps, system2_enabled=False))
    return AgentPolicy(replace(deps, system2_enabled=True))


def run_ladder(n: int, seed: int, deps_factory) -> dict[str, Scorecard]:
    """Run every rung of RUNGS through `run_experiment` and return
    `{rung: Scorecard}`, in RUNGS order.

    `deps_factory(n, seed) -> GraphDeps` is called once per rung (fresh deps:
    a new conn/bandit-prior each rung) so rungs don't leak state into each
    other; `generate_population(n, seed=seed)` is deterministic, so every
    rung's `run_experiment` call still sees the identical CRN population.
    `noaction`/`rules` never touch `deps`, so `deps_factory` may return a
    minimal stub for those (the CLI still builds the real one for all rungs).
    """
    results: dict[str, Scorecard] = {}
    for rung in RUNGS:
        deps = deps_factory(n, seed)
        policy = make_policy(rung, deps)
        results[rung] = run_experiment(policy, n=n, seed=seed)
    return results


def write_scorecards(path: str, ladder: dict) -> None:
    """Write the ladder to THE CONTRACT SCHEMA labs 10-11 read exactly:
    `{"rungs": [{"policy": rung, "scorecard": scorecard.model_dump()}, ...]}`
    in RUNGS order (not dict-insertion order, so callers can rely on it even
    if `ladder` was built out of order).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    order = [r for r in RUNGS if r in ladder] + [r for r in ladder if r not in RUNGS]
    payload = {"rungs": [{"policy": rung, "scorecard": ladder[rung].model_dump()}
                         for rung in order]}
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
