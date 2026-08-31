"""Every tenant-owned table carries TENANT_ID. This test is the guard that a future
migration cannot add a table without one."""
from decimal import Decimal

import pytest
from sqlalchemy import text

from magenta.db import get_conn
from magenta.graph.tables import fulfillment_for, insert_fulfillment
from tests.db_fixtures import TENANT_A

TENANT_TABLES = [
    "GUARDRAIL_CONTACTS",
    "FULFILLMENTS",
    "AUDIT_LOG",
    "BANDIT_POSTERIOR",
    "MEMORY_EDGES",
    "SEMANTIC_CACHE",
]


@pytest.mark.parametrize("table", TENANT_TABLES)
def test_table_exists(migrated_db, table):
    with get_conn() as conn:
        found = conn.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = :t"
            ),
            {"t": table},
        ).scalar()
    assert found == table


@pytest.mark.parametrize("table", TENANT_TABLES)
def test_every_table_has_tenant_id_not_null(migrated_db, table):
    with get_conn() as conn:
        row = conn.execute(
            text(
                "SELECT is_nullable FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = :t "
                "AND column_name = 'TENANT_ID'"
            ),
            {"t": table},
        ).scalar()
    assert row == "NO", f"{table} is missing a NOT NULL TENANT_ID"


def test_fulfillments_pk_is_tenant_scoped(migrated_db):
    """The cross-tenant idempotency collision guard, at the schema level."""
    with get_conn() as conn:
        cols = conn.execute(
            text(
                "SELECT kcu.column_name FROM information_schema.table_constraints tc "
                "JOIN information_schema.key_column_usage kcu "
                "  ON tc.constraint_name = kcu.constraint_name "
                "WHERE tc.table_name = 'FULFILLMENTS' "
                "  AND tc.constraint_type = 'PRIMARY KEY'"
            )
        ).scalars().all()
    assert set(cols) == {"TENANT_ID", "IDEMPOTENCY_KEY"}


def test_organizations_table_exists(migrated_db):
    with get_conn() as conn:
        found = conn.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = 'ORGANIZATIONS'"
            )
        ).scalar()
    assert found == "ORGANIZATIONS"


def test_fulfillment_cost_roundtrips_exactly(db_conn):
    """COST is NUMERIC(12,2), not DOUBLE PRECISION -- money must not come back
    as 8.099999999999999. psycopg returns NUMERIC as decimal.Decimal."""
    row = insert_fulfillment(
        db_conn, TENANT_A, "cost-roundtrip-key", "CUST_0099", "CAMP-1",
        "BILL_CREDIT", 8.10, "FULFILLED",
    )
    assert row["COST"] == Decimal("8.10")

    reread = fulfillment_for(db_conn, TENANT_A, "cost-roundtrip-key")
    assert reread["COST"] == Decimal("8.10")


def test_llm_usage_table_exists(migrated_db):
    with get_conn() as conn:
        cols = conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'LLM_USAGE'"
            )
        ).scalars().all()
    assert {"TENANT_ID", "ROLE", "MODEL", "TOKENS_IN", "TOKENS_OUT", "TS"} <= set(cols)


def test_organizations_has_a_nullable_budget(migrated_db):
    with get_conn() as conn:
        nullable = conn.execute(
            text(
                "SELECT is_nullable FROM information_schema.columns "
                "WHERE table_name = 'ORGANIZATIONS' "
                "AND column_name = 'MONTHLY_TOKEN_BUDGET'"
            )
        ).scalar()
    assert nullable == "YES", "NULL must mean unlimited"


def test_organizations_has_industry_and_contact_columns(migrated_db):
    with get_conn() as conn:
        cols = conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'ORGANIZATIONS'"
            )
        ).scalars().all()
    assert {"INDUSTRY", "ADMIN_CONTACT_EMAIL"} <= set(cols)


def test_industry_column_is_nullable(migrated_db):
    with get_conn() as conn:
        nullable = conn.execute(
            text(
                "SELECT is_nullable FROM information_schema.columns "
                "WHERE table_name = 'ORGANIZATIONS' AND column_name = 'INDUSTRY'"
            )
        ).scalar()
    assert nullable == "YES", "NULL must mean 'profile incomplete'"
