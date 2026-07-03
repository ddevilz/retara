"""Deterministic observable-only feature vectorizer for the risk/uplift models.

ANTI-CIRCULARITY: only fields present on the observable ``Customer`` are used.
Hidden simulator state must never appear here. ``FEATURE_NAMES`` are ALL_CAPS.
"""
from __future__ import annotations

import numpy as np

from magenta.offers import OfferDecision  # noqa: F401  (kept for downstream import parity)
from magenta.sim.population import Customer

NPS_MISSING_SENTINEL = -999.0

# (feature name, extractor). Order is the contract — append only, never reorder.
_NUMERIC_SPECS: list[tuple[str, str]] = [
    ("TENURE_MONTHS", "tenure_months"),
    ("MONTHLY_CHARGE", "monthly_charge"),
    ("TOTAL_CHARGES", "total_charges"),
    ("DATA_GB_USED_P50", "data_gb_used_p50"),
    ("DATA_ALLOWANCE_GB", "data_allowance_gb"),
    ("OVERAGE_EVENTS_90D", "overage_events_90d"),
    ("DROPPED_CALLS_30D", "dropped_calls_30d"),
    ("SUPPORT_TICKETS_90D", "support_tickets_90d"),
    ("LATE_PAYMENTS_12M", "late_payments_12m"),
    ("DEVICE_AGE_MONTHS", "device_age_months"),
    ("CONTRACT_END_DAYS", "contract_end_days"),
    ("GROSS_MARGIN_MONTHLY", "gross_margin_monthly"),
    ("CLV_ESTIMATE", "clv_estimate"),
]

# Derived features (name, callable). Kept after raw numerics.
def _data_util_ratio(c: Customer) -> float:
    allowance = c.data_allowance_gb
    if allowance is None or allowance <= 0:
        return 0.0
    return c.data_gb_used_p50 / allowance


def _charge_per_month_tenure(c: Customer) -> float:
    if c.tenure_months <= 0:
        return float(c.monthly_charge)
    return c.total_charges / c.tenure_months


_DERIVED_SPECS: list[tuple[str, callable]] = [
    ("DATA_UTIL_RATIO", _data_util_ratio),
    ("AVG_CHARGE_PER_TENURE", _charge_per_month_tenure),
]

# NPS handled explicitly (None-safe sentinel).
FEATURE_NAMES: list[str] = (
    [name for name, _ in _NUMERIC_SPECS]
    + ["NPS_LAST"]
    + [name for name, _ in _DERIVED_SPECS]
)


def featurize(c: Customer) -> np.ndarray:
    values: list[float] = [float(getattr(c, attr)) for _, attr in _NUMERIC_SPECS]
    values.append(NPS_MISSING_SENTINEL if c.nps_last is None else float(c.nps_last))
    values.extend(float(fn(c)) for _, fn in _DERIVED_SPECS)
    return np.asarray(values, dtype=np.float64)
