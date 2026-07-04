"""Task 9.3: hard invariant checks behind `@pytest.mark.hardcheck`.

Centralizes the FOUR deterministic compliance invariants (spec §0.5 build
contract) that were previously only asserted piecemeal inside individual
graph node tests:

1. idempotency replay   -- replaying Act never yields more than one
                           FULFILLMENTS row for a customer.
2. holdout purity       -- FULFILLMENTS vs. the audit trail's own
                           holdout/SHADOW record must never agree.
3. guardrail compliance -- every real fulfillment has a matching
                           PASS/NEEDS_APPROVAL GUARDRAIL verdict.
4. no hidden leak       -- L1 latent fields never survive serialization of
                           graph/dialogue state.

No network, no live model artifacts: :memory: sqlite conns / fresh tmp-path
DBs only, and the node exercised here (act) never touches the chat/LLM
client, so nothing needs stubbing. Run with `uv run pytest -m hardcheck`.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from magenta.chat.state import DialogueState
from magenta.evalx.hardchecks import (
    replay_idempotent,
    scan_guardrail_compliance,
    scan_hidden_leak,
    scan_holdout_purity,
)
from magenta.graph.nodes import act
from magenta.graph.state import GuardrailVerdict
from magenta.graph.tables import init_graph_tables
from magenta.offers import Arm, OfferDecision


## --------------------------------------------------------------------------- #
## shared minimal fakes (self-contained -- tests/graph/conftest.py fixtures are
## scoped to tests/graph/, not shared here).
## --------------------------------------------------------------------------- #
class _Deps:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeCustomer:
    customer_id = "CUST-HC1"
    gross_margin_monthly = 22.0


class _FakeCatalog:
    def eligible(self, c):
        return [Arm.BILL_CREDIT]

    def cost(self, arm):
        return 8.0

    def min_margin(self, arm):
        return 0.0


def _mem_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_graph_tables(conn)
    return conn


def _act_state(customer) -> dict:
    return {
        "customer_id": customer.customer_id, "campaign_id": "CAMP-HC",
        "consent_flags": {"MARKETING": True},
        "risk": None, "diagnosis": None,
        "offer": OfferDecision(arm=Arm.BILL_CREDIT, cost=8.0, propensity=0.6),
        "verdict": GuardrailVerdict(decision="PASS"),
        "fulfillment": None, "outcome": None, "messages": [], "audit_log": [],
        "requires_approval": False, "holdout": False,
    }


def _insert_fulfillment_row(conn, key: str, customer_id: str) -> None:
    conn.execute(
        "INSERT INTO FULFILLMENTS "
        "(IDEMPOTENCY_KEY, CUSTOMER_ID, CAMPAIGN_ID, ARM, COST, STATUS) "
        "VALUES (?, ?, 'CAMP', 'BILL_CREDIT', 8.0, 'FULFILLED')",
        (key, customer_id),
    )


def _insert_audit_row(conn, node: str, customer_id: str, payload: dict) -> None:
    conn.execute(
        "INSERT INTO AUDIT_LOG (NODE, CUSTOMER_ID, TS, PAYLOAD) VALUES (?, ?, ?, ?)",
        (node, customer_id, "2026-01-01T00:00:00Z", json.dumps(payload)),
    )


## --------------------------------------------------------------------------- #
## 1. idempotency replay
## --------------------------------------------------------------------------- #
@pytest.mark.hardcheck
def test_replay_idempotent_after_double_act_invoke():
    conn = _mem_conn()
    customer = _FakeCustomer()
    deps = _Deps(load_customer=lambda cid: customer, catalog=_FakeCatalog(), conn=conn)
    s = _act_state(customer)
    act(s, deps)
    act(s, deps)  # replay: simulates interrupt/resume re-running the node
    assert conn.execute("SELECT count(*) FROM FULFILLMENTS").fetchone()[0] == 1
    assert replay_idempotent(conn, customer.customer_id) is True


@pytest.mark.hardcheck
def test_replay_idempotent_detects_duplicate_rows():
    conn = _mem_conn()
    _insert_fulfillment_row(conn, "K1", "C-DUP")
    _insert_fulfillment_row(conn, "K2", "C-DUP")  # simulates a broken dedupe
    conn.commit()
    assert replay_idempotent(conn, "C-DUP") is False


@pytest.mark.hardcheck
def test_replay_idempotent_vacuously_true_on_empty_conn(tmp_path):
    from magenta.db import get_conn

    conn = get_conn(str(tmp_path / "r.db"))
    assert replay_idempotent(conn, "NOBODY") is True


## --------------------------------------------------------------------------- #
## 2. holdout purity: FULFILLMENTS vs. the audit trail's own SHADOW record
## --------------------------------------------------------------------------- #
@pytest.mark.hardcheck
def test_holdout_purity_clean_conn(tmp_path):
    from magenta.db import get_conn

    conn = get_conn(str(tmp_path / "t.db"))
    assert scan_holdout_purity(conn) == []


@pytest.mark.hardcheck
def test_holdout_purity_clean_when_shadow_and_fulfilled_are_disjoint():
    conn = _mem_conn()
    _insert_fulfillment_row(conn, "K1", "C-REAL")
    _insert_audit_row(conn, "ACT", "C-REAL", {"status": "FULFILLED", "arm": "BILL_CREDIT"})
    _insert_audit_row(conn, "ACT", "C-HOLDOUT", {"status": "SHADOW", "arm": "BILL_CREDIT"})
    conn.commit()
    assert scan_holdout_purity(conn) == []


@pytest.mark.hardcheck
def test_holdout_purity_detects_violation():
    conn = _mem_conn()
    # C-LEAK has BOTH a real FULFILLMENTS row AND a SHADOW (holdout) audit
    # entry -- the two signals the graph's act() node keeps mutually
    # exclusive by construction. Their agreement is the purity violation.
    _insert_fulfillment_row(conn, "K1", "C-LEAK")
    _insert_audit_row(conn, "ACT", "C-LEAK", {"status": "SHADOW", "arm": "BILL_CREDIT"})
    conn.commit()
    assert scan_holdout_purity(conn) == ["C-LEAK"]


## --------------------------------------------------------------------------- #
## 3. guardrail compliance: every real fulfillment has a PASS/NEEDS_APPROVAL
## --------------------------------------------------------------------------- #
@pytest.mark.hardcheck
def test_guardrail_compliance_clean_conn(tmp_path):
    from magenta.db import get_conn

    conn = get_conn(str(tmp_path / "g.db"))
    assert scan_guardrail_compliance(conn) == []


@pytest.mark.hardcheck
def test_guardrail_compliance_clean_when_verdict_present():
    conn = _mem_conn()
    _insert_audit_row(conn, "GUARDRAIL", "C-OK", {"decision": "PASS", "failed_policies": []})
    _insert_audit_row(conn, "ACT", "C-OK", {"status": "FULFILLED", "arm": "BILL_CREDIT"})
    conn.commit()
    assert scan_guardrail_compliance(conn) == []


@pytest.mark.hardcheck
def test_guardrail_compliance_detects_violation():
    conn = _mem_conn()
    # C-NOGATE was fulfilled but never has a PASS/NEEDS_APPROVAL GUARDRAIL entry.
    _insert_audit_row(conn, "ACT", "C-NOGATE", {"status": "FULFILLED", "arm": "BILL_CREDIT"})
    conn.commit()
    assert scan_guardrail_compliance(conn) == ["C-NOGATE"]


## --------------------------------------------------------------------------- #
## 4. no hidden leak
## --------------------------------------------------------------------------- #
@pytest.mark.hardcheck
def test_no_hidden_leak_in_dialogue_state():
    st = DialogueState(customer_id="C1", intent_stack=["cancel"], sentiment=-0.2)
    leaks = scan_hidden_leak(st.model_dump())
    assert leaks == []


@pytest.mark.hardcheck
def test_hidden_leak_detects_planted_field():
    poisoned = {"customer_id": "C1", "theta_churn_base": 0.7}
    leaks = scan_hidden_leak(poisoned)
    assert "theta_churn_base" in leaks


@pytest.mark.hardcheck
def test_hidden_leak_detects_field_nested_in_narrative_text():
    # the realistic leak shape in this codebase: a hidden token surfacing
    # inside a free-text field (e.g. an LLM narrative), not a top-level key.
    poisoned = {"customer_id": "C1",
                "narrative": "customer has high competitor_pull and will churn"}
    leaks = scan_hidden_leak(poisoned)
    assert leaks == ["competitor_pull"]
