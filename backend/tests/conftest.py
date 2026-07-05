"""Root-level shared fixtures (Lab 12: memory-wiring tests need a real
GraphDeps, not the loose per-test fakes in tests/graph/conftest.py, so
outcome()'s real featurize() call on a real Customer doesn't blow up)."""
from __future__ import annotations

import sqlite3

import pytest

from magenta.graph.build import GraphDeps
from magenta.graph.state import Diagnosis
from magenta.memory.store import CustomerMemory
from magenta.offers import Arm, OfferDecision
from magenta.sim.oracle import Outcome
from magenta.sim.population import generate_population


class _SpyChat:
    """Minimal chat double: records prompts, returns a canned Diagnosis."""

    def __init__(self, diagnosis: Diagnosis | None = None):
        self.calls: list[dict] = []
        self.prompts: list[str] = []
        self._diagnosis = diagnosis or Diagnosis(
            root_cause_tags=["BILL_SHOCK"],
            narrative="Overage-driven bill shock.",
            eligible_offer_ids=[Arm.BILL_CREDIT.value],
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


class _FakeOracle:
    def __init__(self, accepted=True, churned=False):
        self._accepted, self._churned = accepted, churned

    def outcome(self, customer, offer):
        return Outcome(accepted=self._accepted, churned=self._churned)


class _FakeBandit:
    def __init__(self):
        self.updates: list[tuple] = []

    def update(self, x, arm, reward):
        self.updates.append((arm, reward))

    def save(self, conn):
        pass


class _Params:
    freq_cap_days = 14
    freq_cap_max = 1
    value_cap = 40.0
    p90_clv = 2000.0


@pytest.fixture
def mem_deps_factory():
    """Factory building a real GraphDeps wired with an in-memory
    CustomerMemory (no embedder -- outcome()/diagnose() only need
    add_edge/consolidate/timeline, not semantic_recall), a spy chat, a fake
    oracle/bandit, and a fixed OfferDecision at `deps._offer_fixture`.

    `load_customer` returns a REAL Customer (from generate_population) so
    outcome()'s real featurize() call doesn't AttributeError on a hand-rolled
    stub missing raw fields (total_charges, data_gb_used_p50, ...).
    """

    def _make(**overrides):
        customers, _ = generate_population(1, seed=0)
        customer = customers[0]

        mem_conn = sqlite3.connect(":memory:")
        mem_conn.row_factory = sqlite3.Row
        memory = CustomerMemory(mem_conn)
        memory.init_tables()

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row

        deps = GraphDeps(
            risk=None, uplift=None, bandit=_FakeBandit(), catalog=None,
            oracle=_FakeOracle(), conn=conn, params=_Params(), chat=_SpyChat(),
            load_customer=lambda cid: customer, memory=memory,
        )
        deps._offer_fixture = OfferDecision(
            arm=Arm.BILL_CREDIT, cost=8.0, rationale="test fixture", propensity=0.6)
        for k, v in overrides.items():
            setattr(deps, k, v)
        return deps

    return _make
