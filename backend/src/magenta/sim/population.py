"""L1 hidden latent state + L2 observable Customer + population generator.

ANTI-CIRCULARITY: Customer holds ONLY observable fields (spec §6). Hidden state
(theta_*, segment, competitor_pull) lives in HiddenStore, keyed by customer_id,
and is NEVER a field on Customer. Tests assert the two field-sets are disjoint.
"""

from __future__ import annotations

import json
from enum import Enum

import numpy as np
from pydantic import BaseModel

from magenta.config import data_dir


class Segment(str, Enum):
    PERSUADABLE = "PERSUADABLE"
    SURE_THING = "SURE_THING"
    LOST_CAUSE = "LOST_CAUSE"
    SLEEPING_DOG = "SLEEPING_DOG"


class HiddenState(BaseModel):
    """L1 latent ground truth. Simulator-private. NEVER serialized into agent state."""

    theta_churn_base: float
    theta_price_sens: float
    persuadable_segment: Segment
    competitor_pull: float


class Customer(BaseModel):
    """L2 observable telemetry — the agent's entire world (spec §6)."""

    customer_id: str
    tenure_months: int
    contract: str  # MONTH_TO_MONTH | ONE_YEAR | TWO_YEAR
    monthly_charge: float
    total_charges: float
    plan: str  # BASIC | STANDARD | PREMIUM
    data_gb_used_p50: float
    data_allowance_gb: float
    overage_events_90d: int
    dropped_calls_30d: int
    support_tickets_90d: int
    nps_last: int | None  # ~40% missing
    late_payments_12m: int
    device_age_months: int
    contract_end_days: int
    gross_margin_monthly: float
    clv_estimate: float
    # Telco-seeded demographics (observable)
    senior_citizen: bool
    has_partner: bool
    has_dependents: bool


HiddenStore = dict[str, HiddenState]


def _load_marginals() -> dict:
    with (data_dir() / "telco_marginals.json").open(encoding="utf-8") as fh:
        return json.load(fh)


def _choice(rng: np.random.Generator, dist: dict[str, float]) -> str:
    keys = list(dist)
    probs = np.array([dist[k] for k in keys], dtype=float)
    probs = probs / probs.sum()
    return keys[int(rng.choice(len(keys), p=probs))]


def _sample_segment(rng: np.random.Generator, mix: dict[str, float]) -> Segment:
    return Segment(_choice(rng, mix))


def generate_population(n: int, seed: int) -> tuple[list[Customer], HiddenStore]:
    """Generate n customers + their hidden states. Deterministic in (n, seed)."""
    m = _load_marginals()
    rng = np.random.default_rng(seed)

    customers: list[Customer] = []
    hidden: HiddenStore = {}

    contract_dist = m["contract"]
    plan_dist = m["plan"]
    seg_mix = m["segment_mix"]
    ten = m["tenure_months"]
    charge = m["monthly_charge"]
    allowance = m["data_allowance_gb"]
    nps_missing = m["nps_missing_rate"]

    for i in range(n):
        cid = f"C{i:07d}"

        segment = _sample_segment(rng, seg_mix)
        # hidden latent draws — correlated with, but not revealing, observables
        theta_churn_base = float(np.clip(rng.normal(0.0, 1.0), -3.0, 3.0))
        theta_price_sens = float(np.clip(rng.normal(0.0, 1.0), -3.0, 3.0))
        competitor_pull = float(np.clip(rng.beta(2.0, 5.0), 0.0, 1.0))
        hidden[cid] = HiddenState(
            theta_churn_base=theta_churn_base,
            theta_price_sens=theta_price_sens,
            persuadable_segment=segment,
            competitor_pull=competitor_pull,
        )

        # observables
        plan = _choice(rng, plan_dist)
        contract = _choice(rng, contract_dist)
        tenure = int(np.clip(round(rng.normal(ten["mean"], ten["sd"])),
                             ten["min"], ten["max"]))
        mc_par = charge[plan]
        monthly = float(np.clip(rng.normal(mc_par["mean"], mc_par["sd"]), 15.0, 200.0))
        total = float(round(monthly * tenure * rng.uniform(0.92, 1.05), 2))
        alw = float(allowance[plan])
        # usage: higher churn-risk customers slightly over-use (noisy correlation)
        usage = float(np.clip(
            rng.normal(alw * 0.75, alw * 0.30) + 0.4 * alw * theta_churn_base * 0.1,
            0.1, alw * 2.2,
        ))
        overage = int(max(0, rng.poisson(1.5 if usage > alw else 0.3)))
        dropped = int(max(0, rng.poisson(1.0 + 1.5 * competitor_pull)))
        tickets = int(max(0, rng.poisson(0.8 + 0.6 * max(0.0, theta_churn_base))))
        late = int(max(0, rng.poisson(0.4 + 0.3 * theta_price_sens
                                      if theta_price_sens > 0 else 0.2)))
        nps = None if rng.random() < nps_missing else int(np.clip(
            round(rng.normal(7.0 - theta_churn_base, 2.0)), 0, 10))
        device_age = int(np.clip(round(rng.normal(18.0, 10.0)), 0, 60))
        if contract == "MONTH_TO_MONTH":
            end_days = int(rng.integers(0, 30))
        elif contract == "ONE_YEAR":
            end_days = int(rng.integers(0, 365))
        else:
            end_days = int(rng.integers(0, 730))
        margin = float(round(monthly * rng.uniform(0.35, 0.55), 2))
        clv = float(round(margin * max(1, 60 - device_age % 60)
                          * rng.uniform(0.8, 1.2), 2))

        customers.append(Customer(
            customer_id=cid,
            tenure_months=tenure,
            contract=contract,
            monthly_charge=round(monthly, 2),
            total_charges=total,
            plan=plan,
            data_gb_used_p50=round(usage, 2),
            data_allowance_gb=alw,
            overage_events_90d=overage,
            dropped_calls_30d=dropped,
            support_tickets_90d=tickets,
            nps_last=nps,
            late_payments_12m=late,
            device_age_months=device_age,
            contract_end_days=end_days,
            gross_margin_monthly=margin,
            clv_estimate=clv,
            senior_citizen=bool(rng.random() < 0.16),
            has_partner=bool(rng.random() < 0.48),
            has_dependents=bool(rng.random() < 0.30),
        ))

    return customers, hidden
