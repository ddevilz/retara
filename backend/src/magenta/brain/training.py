"""Generate training labels for the risk and uplift models.

Two INDEPENDENT oracle runs on the SAME population (control uses `seed`, the
randomized-offer run uses `seed + 1` — deliberately NOT CRN-paired; each run
yields valid unbiased labels for its own downstream model):
  1. NO-offer run  -> churn labels for the risk model.
  2. randomized-offer run (~50% treated, random eligible arm) -> uplift labels.
     Note: ~7% of customers have no eligible non-NO_ACTION arm and fall back to
     control, so the realized treated share sits near 0.47, not exactly 0.50.

Hidden state is used ONLY through ``ResponseOracle.outcome``; never as a feature.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from magenta.config import configs_dir
from magenta.offers import Arm, OfferCatalog, OfferDecision
from magenta.sim.oracle import ResponseOracle, SimParams
from magenta.sim.population import Customer, generate_population


@dataclass
class TrainingData:
    customers: list[Customer]
    churned: list[bool]
    treated: list[bool]
    retained: list[bool]
    offers: list[OfferDecision | None]


def _oracle_params() -> SimParams:
    # Params were frozen by labs 0-1; load from sim_params.yaml.
    return SimParams.load(configs_dir() / "sim_params.yaml")


def build_training_data(n: int, seed: int) -> TrainingData:
    customers, hidden = generate_population(n, seed=seed)
    catalog = OfferCatalog.load(configs_dir() / "offers.yaml")
    params = _oracle_params()

    # Run 1: churn labels under no intervention.
    oracle_control = ResponseOracle(hidden, params, seed=seed)
    churned = [bool(oracle_control.outcome(c, None).churned) for c in customers]

    # Run 2: randomized offers -> uplift labels.
    rng = random.Random(seed + 1)
    oracle_treat = ResponseOracle(hidden, params, seed=seed + 1)
    treated: list[bool] = []
    offers: list[OfferDecision | None] = []
    retained: list[bool] = []
    for c in customers:
        is_treated = rng.random() < 0.5
        offer: OfferDecision | None = None
        if is_treated:
            eligible = [a for a in catalog.eligible(c) if a != Arm.NO_ACTION]
            if eligible:
                arm = rng.choice(eligible)
                offer = OfferDecision(
                    arm=arm,
                    cost=catalog.cost(arm),
                    rationale="randomized training offer",
                    propensity=0.5 / max(len(eligible), 1),
                )
            else:
                is_treated = False
        out = oracle_treat.outcome(c, offer)
        treated.append(is_treated)
        offers.append(offer)
        retained.append(not bool(out.churned))

    return TrainingData(
        customers=customers,
        churned=churned,
        treated=treated,
        retained=retained,
        offers=offers,
    )
