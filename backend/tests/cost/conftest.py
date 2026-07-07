"""Shared fixtures for tests/cost (Task 13.4: cache+cascade wired into
diagnose_cohort)."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import pytest

from magenta.brain.risk import Band, Driver
from magenta.brain.uplift import Segment
from magenta.graph.state import RiskUpliftReport, Timing
from magenta.memory.embed import LocalEmbedder


class _Customer:
    def __init__(self, customer_id: str, **obs):
        self.customer_id = customer_id
        self.tenure_months = obs.get("tenure_months", 14)
        self.monthly_charge = obs.get("monthly_charge", 79.0)
        self.overage_events_90d = obs.get("overage_events_90d", 2)
        self.dropped_calls_30d = obs.get("dropped_calls_30d", 0)
        self.support_tickets_90d = obs.get("support_tickets_90d", 1)
        self.contract_end_days = obs.get("contract_end_days", 20)
        self.gross_margin_monthly = obs.get("gross_margin_monthly", 22.0)
        self.clv_estimate = obs.get("clv_estimate", 900.0)


def _report() -> RiskUpliftReport:
    return RiskUpliftReport(
        p_churn=0.7,
        band=Band.HIGH,
        drivers=[Driver(feature="OVERAGE_EVENTS", label="OVERAGE_EVENTS",
                        shap_value=0.3, direction="UP")],
        tau_hat=0.18,
        segment=Segment.PERSUADABLE,
        engage=True,
        timing=Timing.ACT_NOW,
    )


@dataclass
class SmallCohort:
    customers: list
    reports: dict
    deps: object
    conn: sqlite3.Connection
    embedder: object = field(default_factory=LocalEmbedder)


@pytest.fixture(scope="session")
def _shared_embedder():
    # session-scoped: loading the sentence-transformers model is the slow
    # part (see magenta.memory.embed) -- share it across tests/cost/* fixtures.
    return LocalEmbedder()


@pytest.fixture
def small_cohort(_shared_embedder):
    """2 customers with IDENTICAL driver signatures -> the SemanticCache key
    text is byte-identical -> 1 real LLM call, 1 cache hit."""
    customers = [_Customer("CUST-1"), _Customer("CUST-2")]
    reports = {"CUST-1": _report(), "CUST-2": _report()}
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return SmallCohort(customers=customers, reports=reports, deps=object(),
                       conn=conn, embedder=_shared_embedder)
