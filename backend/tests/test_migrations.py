"""Every tenant-owned table carries TENANT_ID. This test is the guard that a future
migration cannot add a table without one."""
import pytest
from sqlalchemy import text

from magenta.db import get_conn

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
