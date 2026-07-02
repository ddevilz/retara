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


def generate_events(
    customer: Customer, hidden: HiddenState, rng: np.random.Generator
) -> list[LifeEvent]:
    """Stochastic life-events for the scoring period. Deterministic given rng state."""
    events: list[LifeEvent] = []

    # OVERAGE — probability rises with recent overage history
    p_overage = min(0.9, 0.08 + 0.12 * customer.overage_events_90d)
    if rng.random() < p_overage:
        events.append(LifeEvent(kind=EventKind.OVERAGE, hazard_multiplier=0.35))

    # DROPPED_CALLS — network pain, correlated with dropped-call history
    p_dropped = min(0.9, 0.05 + 0.10 * customer.dropped_calls_30d)
    if rng.random() < p_dropped:
        events.append(LifeEvent(kind=EventKind.DROPPED_CALLS, hazard_multiplier=0.30))

    # BILL_SHOCK — higher charge + overage → sudden bill jump
    p_shock = min(0.8, 0.04 + 0.05 * customer.overage_events_90d
                  + 0.002 * customer.monthly_charge)
    if rng.random() < p_shock:
        events.append(LifeEvent(kind=EventKind.BILL_SHOCK, hazard_multiplier=0.45))

    # COMPETITOR_OFFER — exogenous, scaled by hidden competitor_pull
    p_comp = 0.05 + 0.25 * hidden.competitor_pull
    if rng.random() < p_comp:
        events.append(LifeEvent(kind=EventKind.COMPETITOR_OFFER, hazard_multiplier=0.50))

    # CONTRACT_EXPIRY — only fires when the contract is actually ending soon
    if customer.contract_end_days <= 45 and rng.random() < 0.6:
        events.append(LifeEvent(kind=EventKind.CONTRACT_EXPIRY, hazard_multiplier=0.55))

    # SERVICE_COMPLAINT — support-ticket pressure
    p_complaint = min(0.85, 0.04 + 0.09 * customer.support_tickets_90d)
    if rng.random() < p_complaint:
        events.append(LifeEvent(kind=EventKind.SERVICE_COMPLAINT, hazard_multiplier=0.40))

    return events
