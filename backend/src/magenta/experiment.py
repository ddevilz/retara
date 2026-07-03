"""Single-period two-arm RCT: treatment vs untouched holdout. Produces the headline
ATE + bootstrap CI (the proof slide). Holdout assignment is a stable 50/50 hash;
the holdout branch NEVER receives an offer (purity). Budget cap = greedy by CLV order.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel

from magenta.config import configs_dir
from magenta.offers import Arm, OfferCatalog, OfferDecision
from magenta.sim.oracle import ResponseOracle, SimParams
from magenta.sim.population import Customer, HiddenStore, Segment, generate_population


@runtime_checkable
class Policy(Protocol):
    def decide(self, c: Customer) -> OfferDecision | None: ...


class NoActionPolicy:
    """Never contacts anyone. The ablation-ladder floor."""

    def decide(self, c: Customer) -> OfferDecision | None:
        return None


class RulesPolicy:
    """Naive baseline: contract ending soon & high charge -> BILL_CREDIT to everyone
    who looks at-risk. Deliberately unselective (sprays offers) — the number the
    full agent must beat."""

    def __init__(self, path=None):
        self.catalog = OfferCatalog.load(path or configs_dir() / "offers.yaml")

    def decide(self, c: Customer) -> OfferDecision | None:
        at_risk = (c.contract_end_days <= 45) or (c.overage_events_90d >= 2) \
            or (c.support_tickets_90d >= 2)
        if not at_risk:
            return None
        off = self.catalog.get(Arm.BILL_CREDIT)
        if Arm.BILL_CREDIT not in self.catalog.eligible(c):
            return None
        return OfferDecision(arm=Arm.BILL_CREDIT, cost=off.cost,
                             rationale="rules: at-risk -> bill credit", propensity=1.0)


class Scorecard(BaseModel):
    churn_treatment: float
    churn_holdout: float
    ate: float
    ci_low: float
    ci_high: float
    n_treatment: int
    n_holdout: int
    offers_made: int
    acceptance_rate: float
    wasted_offer_rate: float
    sleeping_dogs_contacted: int
    euros_retained: float
    offer_spend: float


def assign_holdout(customer_id: str, seed: int) -> bool:
    """Stable 50/50 assignment. True => holdout (never contacted)."""
    h = hashlib.sha256(f"holdout:{customer_id}:{seed}".encode()).digest()
    return (int.from_bytes(h[:8], "big") % 2) == 0


def _run_arms(policy: Policy, n: int, seed: int, budget: float | None) -> list[dict]:
    """Core loop. Returns a per-customer record list (used by scorecard + purity tests)."""
    customers, hidden = generate_population(n, seed=seed)
    params = SimParams.load(configs_dir() / "sim_params.yaml")
    oracle = ResponseOracle(hidden, params, seed=seed)

    # Provisional decisions for the treatment group.
    treated_candidates: list[tuple[Customer, OfferDecision]] = []
    records: list[dict] = []
    for c in customers:
        holdout = assign_holdout(c.customer_id, seed)
        rec = {"customer_id": c.customer_id, "holdout": holdout,
               "offer_arm": None, "accepted": False, "churned": None,
               "spend": 0.0, "segment": hidden[c.customer_id].persuadable_segment}
        if holdout:
            records.append(rec)
            continue
        decision = policy.decide(c)
        if decision is not None:
            treated_candidates.append((c, decision))
        records.append(rec)

    rec_by_id = {r["customer_id"]: r for r in records}

    # Budget cap: greedy by CLV descending until spend would exceed budget.
    if budget is not None:
        treated_candidates.sort(key=lambda t: t[0].clv_estimate, reverse=True)
    spend = 0.0
    chosen_ids: set[str] = set()
    for c, decision in treated_candidates:
        if budget is not None and spend + decision.cost > budget + 1e-9:
            continue
        spend += decision.cost
        chosen_ids.add(c.customer_id)
        rec_by_id[c.customer_id]["offer_arm"] = decision.arm
        rec_by_id[c.customer_id]["spend"] = decision.cost

    # Realize outcomes via the oracle (CRN pairs treatment/holdout at same seed).
    cust_by_id = {c.customer_id: c for c in customers}
    catalog = getattr(policy, "catalog", None) or OfferCatalog.load(
        configs_dir() / "offers.yaml")
    for r in records:
        c = cust_by_id[r["customer_id"]]
        if r["offer_arm"] is None:
            out = oracle.outcome(c, None)
        else:
            off = catalog.get(r["offer_arm"])
            offer_view = _OfferView(off.arm, off.cost, off.fits_causes)
            out = oracle.outcome(c, offer_view)
            r["accepted"] = out.accepted
        r["churned"] = out.churned
    return records


class _OfferView:
    """Duck-typed offer passed to the oracle (arm, cost, fits_causes)."""

    def __init__(self, arm: Arm, cost: float, fits_causes: list[str]):
        self.arm = arm
        self.cost = cost
        self.fits_causes = fits_causes


def _bootstrap_ate_ci(
    treat_churn: np.ndarray, hold_churn: np.ndarray, seed: int, iters: int = 10000
) -> tuple[float, float]:
    rng = np.random.default_rng(seed + 1)
    nt, nh = len(treat_churn), len(hold_churn)
    ates = np.empty(iters)
    for i in range(iters):
        t = treat_churn[rng.integers(0, nt, nt)].mean()
        h = hold_churn[rng.integers(0, nh, nh)].mean()
        ates[i] = h - t  # ate = holdout - treatment
    lo, hi = np.percentile(ates, [2.5, 97.5])
    return float(lo), float(hi)


def run_experiment(
    policy: Policy, n: int, seed: int, budget: float | None = None
) -> Scorecard:
    records = _run_arms(policy, n, seed, budget)

    treat = [r for r in records if not r["holdout"]]
    hold = [r for r in records if r["holdout"]]
    treat_churn = np.array([1.0 if r["churned"] else 0.0 for r in treat])
    hold_churn = np.array([1.0 if r["churned"] else 0.0 for r in hold])

    churn_t = float(treat_churn.mean()) if len(treat_churn) else 0.0
    churn_h = float(hold_churn.mean()) if len(hold_churn) else 0.0
    ate = churn_h - churn_t

    ci_low, ci_high = _bootstrap_ate_ci(treat_churn, hold_churn, seed) \
        if len(treat_churn) and len(hold_churn) else (0.0, 0.0)

    offered = [r for r in treat if r["offer_arm"] is not None]
    offers_made = len(offered)
    accepted = sum(1 for r in offered if r["accepted"])
    acceptance_rate = (accepted / offers_made) if offers_made else 0.0
    # wasted = offer given but customer did not churn even in expectation of no help
    # operational proxy: offered AND not churned AND was a would-stay (accepted=False & stayed)
    wasted = sum(1 for r in offered if not r["accepted"] and not r["churned"])
    wasted_offer_rate = (wasted / offers_made) if offers_made else 0.0
    sleeping_dogs_contacted = sum(
        1 for r in offered if r["segment"] == Segment.SLEEPING_DOG)
    offer_spend = float(sum(r["spend"] for r in treat))

    # euros retained = (churn averted count) * mean monthly margin proxy (single period).
    averted = max(0.0, ate) * len(treat)
    euros_retained = float(averted * 30.0)  # illustrative single-period margin per save

    return Scorecard(
        churn_treatment=churn_t,
        churn_holdout=churn_h,
        ate=ate,
        ci_low=ci_low,
        ci_high=ci_high,
        n_treatment=len(treat),
        n_holdout=len(hold),
        offers_made=offers_made,
        acceptance_rate=acceptance_rate,
        wasted_offer_rate=wasted_offer_rate,
        sleeping_dogs_contacted=sleeping_dogs_contacted,
        euros_retained=euros_retained,
        offer_spend=offer_spend,
    )
