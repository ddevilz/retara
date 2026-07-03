"""L2/L3 boundary: life-event generator. Event intensities are modulated by hidden
state so events CORRELATE with true risk but never REVEAL it. Each event carries a
hazard_multiplier consumed by the oracle's churn logit (sum of multipliers)."""

from __future__ import annotations

from enum import Enum

import numpy as np
from pydantic import BaseModel

from magenta.sim.population import Customer, HiddenState


class EventKind(str, Enum):
    OVERAGE = "OVERAGE"
    DROPPED_CALLS = "DROPPED_CALLS"
    BILL_SHOCK = "BILL_SHOCK"
    COMPETITOR_OFFER = "COMPETITOR_OFFER"
    CONTRACT_EXPIRY = "CONTRACT_EXPIRY"
    SERVICE_COMPLAINT = "SERVICE_COMPLAINT"


class LifeEvent(BaseModel):
    kind: EventKind
    hazard_multiplier: float


# --- observable/hidden -> event-firing-probability coupling ----------------
# calibrated 2026-07-03 per spec §5.6 closed-loop requirement (risk AUC
# 0.80-0.85 band); response params in sim_params.yaml untouched/frozen.
#
# These slopes/intercepts (NOT the hazard_multiplier constants, which stay as
# originally specified) were steepened so that event_sum responds more
# sharply to the OBSERVABLE counts that drive each event. Because
# overage_events_90d / dropped_calls_30d / support_tickets_90d are raw
# features the RiskModel already sees directly (magenta.brain.features),
# widening their effect on event-firing probability adds signal the
# observable-only model can exploit with NO proxy noise, and does not touch
# persuadable_segment's recoverability at all (segment is assigned purely
# from the hidden thetas in population.py, never from events.py).
#
# TRADE-OFF (see population.py's coupling-constants block and the
# "Calibration fix report" in task-3.3-report.md for the full curve): the
# values below are scaled to ~40% of the slope / ~18% of the intercept of
# the configuration that maximized risk AUC in isolation, because that
# configuration pushed the realized base churn rate to ~0.33-0.34 (probability
# caps bind asymmetrically as slopes steepen, biasing the population mean
# upward). This scaled-down setting keeps the base churn rate close to the
# frozen A0's calibrated ~26.5% at the cost of landing risk AUC around
# ~0.75-0.76 instead of the spec's aspirational 0.80-0.85 -- still solidly
# inside the acceptance test's [0.72, 0.92] band.
_P_OVERAGE_BASE = 0.0036
_P_OVERAGE_SLOPE = 0.34
_P_DROPPED_BASE = 0.0072
_P_DROPPED_SLOPE = 0.048
_P_SHOCK_BASE = 0.0036
_P_SHOCK_OVERAGE_SLOPE = 0.12
_P_SHOCK_CHARGE_SLOPE = 0.0025
_P_COMP_BASE = 0.0054
_P_COMP_SLOPE = 0.30
_P_COMPLAINT_BASE = 0.0036
_P_COMPLAINT_SLOPE = 0.24
# CONTRACT_EXPIRY carries the single largest hazard_multiplier (0.55) and its
# eligibility gate (contract_end_days <= 45) covers ~59% of the population
# (all MONTH_TO_MONTH customers are always gate-eligible), yet its firing
# probability was a theta-independent constant -- a large, entirely wasted
# channel. Coupled to theta_churn_base the same way COMPETITOR_OFFER is
# already coupled to hidden.competitor_pull (a hidden-state-scaled
# probability on top of an observable eligibility gate). Base lowered from
# the original 0.6 as part of the churn-rate correction above.
_P_EXPIRY_BASE = 0.108
_P_EXPIRY_CHURN_COEF = 0.24

# Small DIRECT theta_churn_base nudges on OVERAGE/BILL_SHOCK/SERVICE_COMPLAINT
# firing probability, layered on top of their existing observable-count
# terms -- same pattern as COMPETITOR_OFFER's pre-existing direct
# hidden.competitor_pull read and CONTRACT_EXPIRY's above. This channel
# doesn't pass through an extra Poisson-noise layer, so it recovers some of
# the correlation the count-mediated path loses to that noise. Kept modest
# (each event's count-driven term still dominates) to preserve narrative
# sense: theta_churn_base raises how "churn-triggering" a given event reads,
# it doesn't fabricate events out of nothing.
_EVENT_THETA_NUDGE = 0.16


def generate_events(
    customer: Customer, hidden: HiddenState, rng: np.random.Generator
) -> list[LifeEvent]:
    """Stochastic life-events for the scoring period. Deterministic given rng state."""
    events: list[LifeEvent] = []

    # OVERAGE — probability rises with recent overage history (+ small direct
    # theta_churn_base nudge, see _EVENT_THETA_NUDGE)
    p_overage = min(0.95, _P_OVERAGE_BASE + _P_OVERAGE_SLOPE * customer.overage_events_90d
                    + _EVENT_THETA_NUDGE * hidden.theta_churn_base)
    if rng.random() < max(0.0, p_overage):
        events.append(LifeEvent(kind=EventKind.OVERAGE, hazard_multiplier=0.35))

    # DROPPED_CALLS — network pain, correlated with dropped-call history
    p_dropped = min(0.95, _P_DROPPED_BASE + _P_DROPPED_SLOPE * customer.dropped_calls_30d)
    if rng.random() < p_dropped:
        events.append(LifeEvent(kind=EventKind.DROPPED_CALLS, hazard_multiplier=0.30))

    # BILL_SHOCK — higher charge + overage → sudden bill jump (+ small direct
    # theta_churn_base nudge)
    p_shock = min(0.90, _P_SHOCK_BASE + _P_SHOCK_OVERAGE_SLOPE * customer.overage_events_90d
                  + _P_SHOCK_CHARGE_SLOPE * customer.monthly_charge
                  + _EVENT_THETA_NUDGE * hidden.theta_churn_base)
    if rng.random() < max(0.0, p_shock):
        events.append(LifeEvent(kind=EventKind.BILL_SHOCK, hazard_multiplier=0.45))

    # COMPETITOR_OFFER — exogenous, scaled by hidden competitor_pull
    p_comp = min(0.95, _P_COMP_BASE + _P_COMP_SLOPE * hidden.competitor_pull)
    if rng.random() < p_comp:
        events.append(LifeEvent(kind=EventKind.COMPETITOR_OFFER, hazard_multiplier=0.50))

    # CONTRACT_EXPIRY — only fires when the contract is actually ending soon;
    # firing probability (not the eligibility gate) scales with
    # theta_churn_base so higher-risk customers are more likely to treat an
    # ending contract as a churn trigger.
    if customer.contract_end_days <= 45:
        p_expiry = float(np.clip(_P_EXPIRY_BASE + _P_EXPIRY_CHURN_COEF * hidden.theta_churn_base,
                                  0.05, 0.97))
        if rng.random() < p_expiry:
            events.append(LifeEvent(kind=EventKind.CONTRACT_EXPIRY, hazard_multiplier=0.55))

    # SERVICE_COMPLAINT — support-ticket pressure (+ small direct
    # theta_churn_base nudge)
    p_complaint = min(0.92, _P_COMPLAINT_BASE + _P_COMPLAINT_SLOPE * customer.support_tickets_90d
                      + _EVENT_THETA_NUDGE * hidden.theta_churn_base)
    if rng.random() < max(0.0, p_complaint):
        events.append(LifeEvent(kind=EventKind.SERVICE_COMPLAINT, hazard_multiplier=0.40))

    return events
