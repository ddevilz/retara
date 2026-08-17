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

No network, no live model artifacts: the shared `db_conn` Postgres fixture
only (tables truncated after each test), and the node exercised here (act)
never touches the chat/LLM client, so nothing needs stubbing. Run with
`uv run pytest -m hardcheck`.

Every scan is tenant-scoped (Alembic 0001_baseline_schema: FULFILLMENTS'
PK and AUDIT_LOG both carry TENANT_ID) -- tests exercise cross-tenant
isolation directly, not just the single-tenant happy path.
"""
from __future__ import annotations

import json

import pytest
from sqlalchemy import text

from magenta.chat.state import DialogueState
from magenta.evalx.hardchecks import (
    _table_exists,
    replay_idempotent,
    scan_guardrail_compliance,
    scan_hidden_leak,
    scan_holdout_purity,
)
from magenta.graph.nodes import act
from magenta.graph.state import GuardrailVerdict
from magenta.offers import Arm, OfferDecision
from tests.db_fixtures import TENANT_A, TENANT_B


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


def _act_state(customer, tenant_id) -> dict:
    return {
        "customer_id": customer.customer_id, "campaign_id": "CAMP-HC",
        "tenant_id": tenant_id,
        "consent_flags": {"MARKETING": True},
        "risk": None, "diagnosis": None,
        "offer": OfferDecision(arm=Arm.BILL_CREDIT, cost=8.0, propensity=0.6),
        "verdict": GuardrailVerdict(decision="PASS"),
        "fulfillment": None, "outcome": None, "messages": [], "audit_log": [],
        "requires_approval": False, "holdout": False,
    }


def _insert_fulfillment_row(conn, tenant_id: str, key: str, customer_id: str) -> None:
    conn.execute(
        text(
            'INSERT INTO "FULFILLMENTS" '
            '("TENANT_ID", "IDEMPOTENCY_KEY", "CUSTOMER_ID", "CAMPAIGN_ID", "ARM", "COST", "STATUS") '
            "VALUES (:tenant_id, :key, :customer_id, 'CAMP', 'BILL_CREDIT', 8.0, 'FULFILLED')"
        ),
        {"tenant_id": tenant_id, "key": key, "customer_id": customer_id},
    )


def _insert_audit_row(conn, tenant_id: str, node: str, customer_id: str, payload: dict) -> None:
    conn.execute(
        text(
            'INSERT INTO "AUDIT_LOG" ("TENANT_ID", "NODE", "CUSTOMER_ID", "TS", "PAYLOAD") '
            "VALUES (:tenant_id, :node, :customer_id, :ts, CAST(:payload AS jsonb))"
        ),
        {"tenant_id": tenant_id, "node": node, "customer_id": customer_id,
         "ts": "2026-01-01T00:00:00Z", "payload": json.dumps(payload)},
    )


## --------------------------------------------------------------------------- #
## 0. _table_exists -- the information_schema swap-in for sqlite_master
## --------------------------------------------------------------------------- #
@pytest.mark.hardcheck
def test_table_exists_uses_information_schema(db_conn):
    """sqlite_master does not exist on Postgres."""
    assert _table_exists(db_conn, "FULFILLMENTS") is True
    assert _table_exists(db_conn, "NO_SUCH_TABLE") is False


@pytest.mark.hardcheck
def test_missing_table_branch_of_scans(db_conn):
    """Restores coverage of the `not _table_exists(...)` guard branch.

    Task 3's sweep to the shared Postgres `db_conn` fixture made every table
    always exist (merely empty), so the "missing table" guards in
    scan_holdout_purity/scan_guardrail_compliance/replay_idempotent went
    uncovered. Drop FULFILLMENTS and AUDIT_LOG within this test's connection
    (never committed -- db_conn truncates+reconnects on teardown, so the drop
    never escapes this test) to exercise the FALSE branch directly.
    """
    db_conn.execute(text('DROP TABLE "FULFILLMENTS"'))
    db_conn.execute(text('DROP TABLE "AUDIT_LOG"'))
    assert _table_exists(db_conn, "FULFILLMENTS") is False
    assert _table_exists(db_conn, "AUDIT_LOG") is False

    assert scan_holdout_purity(db_conn, TENANT_A) == []
    assert scan_guardrail_compliance(db_conn, TENANT_A) == []
    assert replay_idempotent(db_conn, TENANT_A, "NOBODY") is True


## --------------------------------------------------------------------------- #
## 1. idempotency replay
## --------------------------------------------------------------------------- #
@pytest.mark.hardcheck
def test_replay_idempotent_after_double_act_invoke(db_conn):
    customer = _FakeCustomer()
    deps = _Deps(load_customer=lambda cid: customer, catalog=_FakeCatalog(),
                 conn=db_conn, tenant_id=TENANT_A)
    s = _act_state(customer, TENANT_A)
    act(s, deps)
    act(s, deps)  # replay: simulates interrupt/resume re-running the node
    db_conn.commit()
    assert db_conn.execute(
        text('SELECT count(*) FROM "FULFILLMENTS" WHERE "TENANT_ID" = :t'),
        {"t": TENANT_A},
    ).scalar_one() == 1
    assert replay_idempotent(db_conn, TENANT_A, customer.customer_id) is True


@pytest.mark.hardcheck
def test_replay_idempotent_detects_duplicate_rows(db_conn):
    _insert_fulfillment_row(db_conn, TENANT_A, "K1", "C-DUP")
    _insert_fulfillment_row(db_conn, TENANT_A, "K2", "C-DUP")  # simulates a broken dedupe
    db_conn.commit()
    assert replay_idempotent(db_conn, TENANT_A, "C-DUP") is False


@pytest.mark.hardcheck
def test_replay_idempotent_vacuously_true_on_empty_conn(db_conn):
    # FULFILLMENTS already exists via the Alembic migration and db_conn starts
    # each test with an empty table -- no rows for this tenant, no init needed.
    assert replay_idempotent(db_conn, TENANT_A, "NOBODY") is True


@pytest.mark.hardcheck
def test_replay_idempotent_is_tenant_scoped(db_conn):
    """Two tenants can each hold one row under the SAME customer_id without
    tripping each other's idempotency check -- FULFILLMENTS' PK is
    (TENANT_ID, IDEMPOTENCY_KEY), not just IDEMPOTENCY_KEY."""
    _insert_fulfillment_row(db_conn, TENANT_A, "KA", "C-SHARED")
    _insert_fulfillment_row(db_conn, TENANT_B, "KB", "C-SHARED")
    db_conn.commit()
    assert replay_idempotent(db_conn, TENANT_A, "C-SHARED") is True
    assert replay_idempotent(db_conn, TENANT_B, "C-SHARED") is True


## --------------------------------------------------------------------------- #
## 2. holdout purity: FULFILLMENTS vs. the audit trail's own SHADOW record
## --------------------------------------------------------------------------- #
@pytest.mark.hardcheck
def test_holdout_purity_clean_conn(db_conn):
    assert scan_holdout_purity(db_conn, TENANT_A) == []


@pytest.mark.hardcheck
def test_holdout_purity_clean_when_shadow_and_fulfilled_are_disjoint(db_conn):
    _insert_fulfillment_row(db_conn, TENANT_A, "K1", "C-REAL")
    _insert_audit_row(db_conn, TENANT_A, "ACT", "C-REAL", {"status": "FULFILLED", "arm": "BILL_CREDIT"})
    _insert_audit_row(db_conn, TENANT_A, "ACT", "C-HOLDOUT", {"status": "SHADOW", "arm": "BILL_CREDIT"})
    db_conn.commit()
    assert scan_holdout_purity(db_conn, TENANT_A) == []


@pytest.mark.hardcheck
def test_holdout_purity_detects_violation(db_conn):
    # C-LEAK has BOTH a real FULFILLMENTS row AND a SHADOW (holdout) audit
    # entry -- the two signals the graph's act() node keeps mutually
    # exclusive by construction. Their agreement is the purity violation.
    _insert_fulfillment_row(db_conn, TENANT_A, "K1", "C-LEAK")
    _insert_audit_row(db_conn, TENANT_A, "ACT", "C-LEAK", {"status": "SHADOW", "arm": "BILL_CREDIT"})
    db_conn.commit()
    assert scan_holdout_purity(db_conn, TENANT_A) == ["C-LEAK"]


@pytest.mark.hardcheck
def test_holdout_purity_is_tenant_scoped(db_conn):
    """A violation in tenant B must never surface in tenant A's scan."""
    _insert_fulfillment_row(db_conn, TENANT_B, "K1", "C-LEAK")
    _insert_audit_row(db_conn, TENANT_B, "ACT", "C-LEAK", {"status": "SHADOW", "arm": "BILL_CREDIT"})
    db_conn.commit()
    assert scan_holdout_purity(db_conn, TENANT_A) == []
    assert scan_holdout_purity(db_conn, TENANT_B) == ["C-LEAK"]


## --------------------------------------------------------------------------- #
## 3. guardrail compliance: every real fulfillment has a PASS/NEEDS_APPROVAL
## --------------------------------------------------------------------------- #
@pytest.mark.hardcheck
def test_guardrail_compliance_clean_conn(db_conn):
    assert scan_guardrail_compliance(db_conn, TENANT_A) == []


@pytest.mark.hardcheck
def test_guardrail_compliance_clean_when_verdict_present(db_conn):
    _insert_audit_row(db_conn, TENANT_A, "GUARDRAIL", "C-OK", {"decision": "PASS", "failed_policies": []})
    _insert_audit_row(db_conn, TENANT_A, "ACT", "C-OK", {"status": "FULFILLED", "arm": "BILL_CREDIT"})
    db_conn.commit()
    assert scan_guardrail_compliance(db_conn, TENANT_A) == []


@pytest.mark.hardcheck
def test_guardrail_compliance_detects_violation(db_conn):
    # C-NOGATE was fulfilled but never has a PASS/NEEDS_APPROVAL GUARDRAIL entry.
    _insert_audit_row(db_conn, TENANT_A, "ACT", "C-NOGATE", {"status": "FULFILLED", "arm": "BILL_CREDIT"})
    db_conn.commit()
    assert scan_guardrail_compliance(db_conn, TENANT_A) == ["C-NOGATE"]


@pytest.mark.hardcheck
def test_guardrail_compliance_is_tenant_scoped(db_conn):
    _insert_audit_row(db_conn, TENANT_B, "ACT", "C-NOGATE", {"status": "FULFILLED", "arm": "BILL_CREDIT"})
    db_conn.commit()
    assert scan_guardrail_compliance(db_conn, TENANT_A) == []
    assert scan_guardrail_compliance(db_conn, TENANT_B) == ["C-NOGATE"]


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
