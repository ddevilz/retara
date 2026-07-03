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


# --- persuadable_segment: probabilistic assignment FROM HIDDEN STATE --------
#
# Spec requirement: segment must NOT be an independent draw from the static
# mix (that caps any uplift-targeting model at random by construction, since
# persuadability would carry zero signal in observables X). Instead it is
# sampled from a softmax over per-segment logits built from the
# already-sampled hidden thetas, so it inherits the same weak,
# noisy observable-correlation that the thetas themselves have — real
# randomness (a categorical draw over calibrated probabilities), never a
# deterministic threshold.
#
# Directional logic baked into the weights below:
#   PERSUADABLE:   higher theta_churn_base AND higher theta_price_sens
#                  (their churn risk is price-fixable).
#   LOST_CAUSE:    higher theta_churn_base + high competitor_pull + LOW
#                  theta_price_sens (leaving regardless; offers can't help).
#   SURE_THING:    low theta_churn_base (nothing to fix — they weren't
#                  leaving anyway).
#   SLEEPING_DOG:  low-to-mid churn risk (|theta_churn_base| small) AND low
#                  theta_price_sens as a contact-aversion proxy (not
#                  deal-driven — a retention outreach reads as unwanted
#                  attention rather than a fix).
#
# _SEG_INTERCEPT values were calibrated offline (fixed-point iteration:
# a_i += log(target_i) - log(achieved_i), integrating over the theta priors
# used in generate_population, 400k-sample Monte Carlo) so the *expected*
# mix reproduces the spec target of 25/50/17/8 exactly. If the target mix in
# telco_marginals.json ever changes, or the weights below change, these
# intercepts need to be recalibrated the same way.
_SEG_WEIGHTS = {
    "PERSUADABLE": {"churn": 0.55, "price": 0.55},
    "SURE_THING": {"churn": -0.85},
    "LOST_CAUSE": {"churn": 0.45, "competitor": 1.1, "price": -0.55},
    "SLEEPING_DOG": {"abs_churn": -0.55, "price": -0.45},
}
_SEG_INTERCEPT = {
    "PERSUADABLE": -0.1491,
    "SURE_THING": 0.7407,
    "LOST_CAUSE": -0.8227,
    "SLEEPING_DOG": -0.6973,
}
_SEG_ORDER = [Segment.PERSUADABLE, Segment.SURE_THING, Segment.LOST_CAUSE, Segment.SLEEPING_DOG]


# --- hidden -> observable coupling constants --------------------------------
# calibrated 2026-07-03 per spec §5.6 closed-loop requirement (risk AUC
# 0.80-0.85 band); response params in sim_params.yaml untouched/frozen.
#
# Diagnosis (see .superpowers/sdd/task-3.3-report.md): the oracle's churn
# logit is dominated by the DIRECT theta_churn_base term (A1=0.85 on a
# unit-variance draw), not by event_sum or competitor_pull (A3=0.90 on a
# Beta(2,5) draw with variance ~0.026). An observable-only model can only
# approach the oracle's discriminative ceiling if theta_churn_base (and, to a
# much smaller extent, competitor_pull) is well-reflected in the observable
# fields it trains on. These constants raise that reflection strength.
# theta_price_sens is deliberately left weakly coupled (only late_payments)
# -- it does not enter the churn oracle at all, so amplifying it would
# inflate persuadable_segment's observable recoverability (spec anti-
# circularity band: 0.56-0.85) without buying any risk-AUC headroom.
#
# TRADE-OFF FOUND (see "Calibration fix report" appended to
# task-3.3-report.md for the full curve): pushing these couplings hard
# enough to land risk AUC in [0.78, 0.88] structurally drags the realized
# base churn rate up to ~0.33-0.34 (vs. the frozen A0's calibrated ~0.265),
# because every event-firing-probability channel is bounded in [0, cap] --
# steepening a channel's response to theta_churn_base saturates the cap for
# the theta>0 half of the population before it saturates the floor for the
# theta<0 half, biasing the population MEAN upward as a side effect of
# widening the SPREAD. That bias cannot be fully offset by lowering
# intercepts alone (verified: even near-zero intercepts only bring the mean
# to ~0.30-0.31 once slopes are steep enough for AUC~0.78). We chose to
# preserve churn~26.5% (a spec target with an explicit CLI verification
# command) over the full risk-AUC aim, landing at AUC~0.75-0.76 (n=6k/3k;
# comfortably inside the actual failing test's [0.72, 0.92] band, short of
# the tighter [0.78, 0.88] internal aim). All constants below are scaled to
# ~40% of the slope / ~18% of the intercept of the "AUC-maximizing, ignore
# churn-rate" configuration that was measured during tuning.
#
# Usage <- theta_churn_base: linear in theta (zero-mean over the population,
# so the marginal mean data_gb_used_p50 is preserved) with reduced Gaussian
# noise SD to raise the theta signal's share of total variance.
_USAGE_MEAN_FRAC = 0.75
_USAGE_NOISE_SD_FRAC = 0.16
_USAGE_CHURN_COEF = 0.52

# Overage events (90d) <- usage/allowance ratio (itself theta_churn_base-
# coupled above): continuous Poisson rate instead of a binary threshold, so
# the graded signal in `usage` isn't discarded.
_OVERAGE_RATE_BASE = 0.054
_OVERAGE_RATE_SLOPE = 7.2

# Support tickets (90d) <- theta_churn_base: two-sided (was one-sided
# max(0, theta)) Poisson rate, floored above zero. Two-sided roughly
# preserves the population mean ticket count (theta is zero-mean) while
# substantially increasing the signal slope. theta_churn_base is the
# DOMINANT term in the churn oracle (A1=0.85, vs A3=0.90 on competitor_pull's
# tiny Beta(2,5) variance) so this and the overage/NPS channels above/below
# carry most of the achievable observable signal -- see dropped_calls below
# for why the competitor_pull channel is deliberately NOT pushed as hard.
_TICKETS_BASE = 0.099
_TICKETS_CHURN_COEF = 7.2
_TICKETS_MIN_LAMBDA = 0.05

# Dropped calls (30d) <- competitor_pull: kept modest on purpose. Pushing
# this channel harder (tried up to coef=10.5) saturated p_dropped's firing
# probability for ~70% of the population regardless of pull -- because
# competitor_pull's own population variance is tiny (Beta(2,5) Var~0.026),
# a Poisson-count proxy that reaches saturation this easily is spending
# churn-rate budget on noise, not signal. Left near the original scale, with
# only a mild bump.
_DROPPED_BASE = 0.108
_DROPPED_COMPETITOR_COEF = 2.2

# Last NPS <- theta_churn_base: stronger coefficient + tighter noise SD.
# ~40% missingness (independent of theta) is unchanged. NOT scaled down with
# the churn-rate correction above -- NPS never feeds churn_prob/event_sum
# (it's a RiskModel-only feature), so it is a "free" theta_churn_base proxy
# channel with zero base-churn-rate cost. Kept at full strength.
_NPS_CHURN_COEF = 5.5
_NPS_NOISE_SD = 0.85

# Late payments (12m): tried adding a theta_churn_base term here too (it
# doesn't feed churn_prob/event_sum, so -- like NPS -- it would have been a
# free proxy channel). Reverted: because `rng.poisson`'s variate algorithm
# consumes a lambda-dependent amount of the underlying bitstream, changing
# this lambda shifts every subsequent rng draw for the *rest of the
# population* -- an uncontrolled reshuffle, not an additive signal. Measured
# net effect across 5 seed pairs was negative (mean risk AUC 0.727 vs 0.740
# without it) purely from that reshuffle, not from any real information
# content. Left theta_price_sens-only, as originally.


def _sample_segment(
    rng: np.random.Generator,
    theta_churn_base: float,
    theta_price_sens: float,
    competitor_pull: float,
) -> Segment:
    w = _SEG_WEIGHTS
    logits = np.array([
        _SEG_INTERCEPT["PERSUADABLE"]
        + w["PERSUADABLE"]["churn"] * theta_churn_base
        + w["PERSUADABLE"]["price"] * theta_price_sens,
        _SEG_INTERCEPT["SURE_THING"]
        + w["SURE_THING"]["churn"] * theta_churn_base,
        _SEG_INTERCEPT["LOST_CAUSE"]
        + w["LOST_CAUSE"]["churn"] * theta_churn_base
        + w["LOST_CAUSE"]["competitor"] * competitor_pull
        + w["LOST_CAUSE"]["price"] * theta_price_sens,
        _SEG_INTERCEPT["SLEEPING_DOG"]
        + w["SLEEPING_DOG"]["abs_churn"] * abs(theta_churn_base)
        + w["SLEEPING_DOG"]["price"] * theta_price_sens,
    ])
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    return _SEG_ORDER[int(rng.choice(len(_SEG_ORDER), p=probs))]


def generate_population(n: int, seed: int) -> tuple[list[Customer], HiddenStore]:
    """Generate n customers + their hidden states. Deterministic in (n, seed)."""
    m = _load_marginals()
    rng = np.random.default_rng(seed)

    customers: list[Customer] = []
    hidden: HiddenStore = {}

    contract_dist = m["contract"]
    plan_dist = m["plan"]
    ten = m["tenure_months"]
    charge = m["monthly_charge"]
    allowance = m["data_allowance_gb"]
    nps_missing = m["nps_missing_rate"]

    for i in range(n):
        cid = f"C{i:07d}"

        # hidden latent draws — correlated with, but not revealing, observables
        theta_churn_base = float(np.clip(rng.normal(0.0, 1.0), -3.0, 3.0))
        theta_price_sens = float(np.clip(rng.normal(0.0, 1.0), -3.0, 3.0))
        competitor_pull = float(np.clip(rng.beta(2.0, 5.0), 0.0, 1.0))
        # segment is assigned PROBABILISTICALLY FROM HIDDEN STATE (spec), not
        # as an independent draw — see _sample_segment for the mechanism.
        segment = _sample_segment(rng, theta_churn_base, theta_price_sens, competitor_pull)
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
        # usage: higher churn-risk customers over-use more (noisy correlation)
        usage = float(np.clip(
            rng.normal(alw * _USAGE_MEAN_FRAC, alw * _USAGE_NOISE_SD_FRAC)
            + alw * _USAGE_CHURN_COEF * theta_churn_base,
            0.1, alw * 2.2,
        ))
        # continuous in the usage/allowance ratio (was a coarse binary
        # threshold) so overage_events_90d carries graded, not just
        # thresholded, theta_churn_base signal via `usage`.
        overage_rate = float(np.clip(
            _OVERAGE_RATE_BASE + _OVERAGE_RATE_SLOPE * (usage / alw - 1.0),
            0.05, 6.0,
        ))
        overage = int(max(0, rng.poisson(overage_rate)))
        dropped = int(max(0, rng.poisson(_DROPPED_BASE
                                         + _DROPPED_COMPETITOR_COEF * competitor_pull)))
        tickets = int(max(0, rng.poisson(max(
            _TICKETS_MIN_LAMBDA, _TICKETS_BASE + _TICKETS_CHURN_COEF * theta_churn_base
        ))))
        late = int(max(0, rng.poisson(0.4 + 0.3 * theta_price_sens
                                      if theta_price_sens > 0 else 0.2)))
        nps = None if rng.random() < nps_missing else int(np.clip(
            round(rng.normal(7.0 - _NPS_CHURN_COEF * theta_churn_base, _NPS_NOISE_SD)), 0, 10))
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
