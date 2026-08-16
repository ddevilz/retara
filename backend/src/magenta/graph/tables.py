"""Tenant-scoped queries for the Guardrail, Act, and audit-trail nodes.

Schema lives in Alembic (`0001_baseline_schema`), NOT here — the old
`init_graph_tables` CREATE TABLE statements are gone.

Every function takes `tenant_id` as its first argument after `conn`. That is not
decoration: FULFILLMENTS' primary key is (TENANT_ID, IDEMPOTENCY_KEY), and a query
that forgets the tenant reads another tenant's data.
"""
from __future__ import annotations

import hashlib
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.engine import Connection

from magenta.offers import Arm

# Placeholder tenant for single-tenant callers (CLI, demo API) until Clerk auth
# lands in Phase 1.2 and resolves a real organisation per request.
DEFAULT_TENANT_ID = "org_default"


def idempotency_key(tenant_id: str, customer_id: str, campaign_id: str, arm: Arm) -> str:
    """TENANT_ID is part of the key. Without it, two tenants using the same customer
    ID format (CUST_0001) collide and the second tenant's genuine offer is silently
    suppressed as a duplicate."""
    raw = f"{tenant_id}:{customer_id}:{campaign_id}:{arm.value}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def record_contact(conn: Connection, tenant_id: str, customer_id: str,
                   campaign_id: str, contacted_at: datetime) -> None:
    conn.execute(
        text(
            'INSERT INTO "GUARDRAIL_CONTACTS" '
            '("TENANT_ID", "CUSTOMER_ID", "CAMPAIGN_ID", "CONTACTED_AT") '
            "VALUES (:tenant_id, :customer_id, :campaign_id, :contacted_at)"
        ),
        {
            "tenant_id": tenant_id,
            "customer_id": customer_id,
            "campaign_id": campaign_id,
            "contacted_at": contacted_at,
        },
    )
    conn.commit()


def contacts_since(conn: Connection, tenant_id: str, customer_id: str,
                   since: datetime) -> int:
    return conn.execute(
        text(
            'SELECT count(*) FROM "GUARDRAIL_CONTACTS" '
            'WHERE "TENANT_ID" = :tenant_id AND "CUSTOMER_ID" = :customer_id '
            'AND "CONTACTED_AT" >= :since'
        ),
        {"tenant_id": tenant_id, "customer_id": customer_id, "since": since},
    ).scalar_one()


def fulfillment_for(conn: Connection, tenant_id: str,
                    idempotency_key: str) -> dict | None:
    row = conn.execute(
        text(
            'SELECT * FROM "FULFILLMENTS" '
            'WHERE "TENANT_ID" = :tenant_id AND "IDEMPOTENCY_KEY" = :key'
        ),
        {"tenant_id": tenant_id, "key": idempotency_key},
    ).mappings().first()
    return dict(row) if row is not None else None


def insert_fulfillment(conn: Connection, tenant_id: str, idempotency_key: str,
                       customer_id: str, campaign_id: str, arm: str,
                       cost: float, status: str) -> dict:
    """Insert-or-return. `ON CONFLICT DO NOTHING` replaces the old
    read-then-insert-then-catch-IntegrityError dance: one statement, no race window,
    and no SQLite-specific exception class to catch."""
    conn.execute(
        text(
            'INSERT INTO "FULFILLMENTS" '
            '("TENANT_ID", "IDEMPOTENCY_KEY", "CUSTOMER_ID", "CAMPAIGN_ID", '
            ' "ARM", "COST", "STATUS") '
            "VALUES (:tenant_id, :key, :customer_id, :campaign_id, "
            "        :arm, :cost, :status) "
            'ON CONFLICT ("TENANT_ID", "IDEMPOTENCY_KEY") DO NOTHING'
        ),
        {
            "tenant_id": tenant_id,
            "key": idempotency_key,
            "customer_id": customer_id,
            "campaign_id": campaign_id,
            "arm": arm,
            "cost": cost,
            "status": status,
        },
    )
    conn.commit()
    winner = fulfillment_for(conn, tenant_id, idempotency_key)
    if winner is None:
        raise RuntimeError(
            f"FULFILLMENTS row vanished after insert for {tenant_id}:{idempotency_key}"
        )
    return winner
