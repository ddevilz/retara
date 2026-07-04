"""Shared fakes + fixtures for graph tests.

Fakes mirror the labs 0-5 interfaces exactly (duck-typed) so nodes never
touch a real model, oracle, or DB during unit tests. `SpyChat` records every
prompt string so we can assert no hidden-state leak.
"""
import numpy as np
import pytest

from magenta.brain.risk import Band, Driver
from magenta.brain.uplift import Segment
from magenta.graph.state import Diagnosis, RiskUpliftReport, Timing
from magenta.offers import Arm, OfferDecision


## ---- observable-only customer stub (matches magenta.sim Customer surface) ----
class FakeCustomer:
    def __init__(self, customer_id="CUST-1", **obs):
        self.customer_id = customer_id
        # a handful of L2 observable fields nodes/prompts may read
        self.tenure_months = obs.get("tenure_months", 14)
        self.monthly_charge = obs.get("monthly_charge", 79.0)
        self.overage_events_90d = obs.get("overage_events_90d", 2)
        self.dropped_calls_30d = obs.get("dropped_calls_30d", 3)
        self.support_tickets_90d = obs.get("support_tickets_90d", 1)
        self.gross_margin_monthly = obs.get("gross_margin_monthly", 22.0)
        self.clv_estimate = obs.get("clv_estimate", 900.0)
        self.contract_end_days = obs.get("contract_end_days", 20)


@pytest.fixture
def customer():
    return FakeCustomer()


## ---- spy LLM: records prompts, returns a canned structured object ----------
class SpyChat:
    def __init__(self, diagnosis: Diagnosis | None = None):
        self.calls: list[dict] = []
        self.prompts: list[str] = []
        self._diagnosis = diagnosis or Diagnosis(
            root_cause_tags=["BILL_SHOCK"],
            narrative="Overage-driven bill shock.",
            eligible_offer_ids=[Arm.BILL_CREDIT.value, Arm.PLAN_DOWNSELL.value],
            confidence=0.8,
        )

    def chat_structured(self, role, messages, model_cls):
        self.calls.append({"role": role, "messages": messages, "model_cls": model_cls})
        for m in messages:
            self.prompts.append(str(m.get("content", "")))
        return self._diagnosis

    def chat(self, role, messages, **kw):
        self.calls.append({"role": role, "messages": messages, "kw": kw})
        for m in messages:
            self.prompts.append(str(m.get("content", "")))
        return "ok"


@pytest.fixture
def spy_chat():
    return SpyChat()


## ---- fake ML brain ---------------------------------------------------------
class FakeRisk:
    def __init__(self, p_churn=0.72, band=Band.HIGH):
        self._p, self._band = p_churn, band

    def score(self, c):
        from magenta.brain.risk import RiskAssessment

        return RiskAssessment(
            p_churn=self._p,
            band=self._band,
            drivers=[
                Driver(feature="OVERAGE_EVENTS", label="Overage events",
                       shap_value=0.31, direction="UP"),
                Driver(feature="TENURE_MONTHS", label="Tenure",
                       shap_value=-0.12, direction="DOWN"),
            ],
        )


class FakeUplift:
    def __init__(self, tau=0.18, segment=Segment.PERSUADABLE):
        self._tau, self._seg = tau, segment

    def tau(self, c):
        return self._tau


class FakeBandit:
    def __init__(self, arm=Arm.BILL_CREDIT, propensity=0.6):
        self._arm, self._p = arm, propensity
        self.updates: list[tuple] = []

    def select(self, x, eligible):
        arm = self._arm if self._arm in eligible else eligible[0]
        return arm, self._p

    def update(self, x, arm, reward):
        self.updates.append((arm, reward))

    def save(self, conn):
        pass


class FakeCatalog:
    def __init__(self, cost=8.0, min_margin=5.0):
        self._cost, self._min_margin = cost, min_margin

    def eligible(self, c):
        return [Arm.BILL_CREDIT, Arm.PLAN_DOWNSELL, Arm.DATA_BOOST]

    def cost(self, arm):
        return self._cost

    def min_margin(self, arm):
        return self._min_margin


class FakeOracle:
    def __init__(self, accepted=True, churned=False):
        self._acc, self._churn = accepted, churned

    def outcome(self, customer, offer):
        from magenta.sim.oracle import Outcome  # real dataclass/model

        return Outcome(accepted=self._acc, churned=self._churn)


@pytest.fixture
def fakes():
    return {
        "risk": FakeRisk(),
        "uplift": FakeUplift(),
        "bandit": FakeBandit(),
        "catalog": FakeCatalog(),
        "oracle": FakeOracle(),
    }


def _report(engage=True, segment=Segment.PERSUADABLE):
    return RiskUpliftReport(
        p_churn=0.72, band=Band.HIGH,
        drivers=[Driver(feature="OVERAGE_EVENTS", label="Overage events",
                        shap_value=0.31, direction="UP")],
        tau_hat=0.18, segment=segment, engage=engage, timing=Timing.ACT_NOW,
    )
