"""SQLite tables backing the Guardrail, Act, and audit-trail nodes.

- GUARDRAIL_CONTACTS: frequency-cap ledger (§5.7). One row per proactive contact.
- FULFILLMENTS: the idempotency ledger (§5.5 Act). UNIQUE on IDEMPOTENCY_KEY
  guarantees exactly-once fulfill even when a node re-runs after interrupt/resume.
- AUDIT_LOG: one row per executed node per graph run (§5.5 / Task 6.6), flushed
  by `magenta.graph.build.persist_audit` after `graph.invoke(...)` returns.

All column names ALL_CAPS (CLAUDE.md convention; Postgres-fold-safe).
"""
from __future__ import annotations

import sqlite3

_DDL_CONTACTS = """
CREATE TABLE IF NOT EXISTS GUARDRAIL_CONTACTS (
    ID           INTEGER PRIMARY KEY AUTOINCREMENT,
    CUSTOMER_ID  TEXT NOT NULL,
    CAMPAIGN_ID  TEXT NOT NULL,
    CONTACTED_AT TEXT NOT NULL
)
"""

_DDL_FULFILLMENTS = """
CREATE TABLE IF NOT EXISTS FULFILLMENTS (
    IDEMPOTENCY_KEY TEXT PRIMARY KEY,
    CUSTOMER_ID     TEXT NOT NULL,
    CAMPAIGN_ID     TEXT NOT NULL,
    ARM             TEXT NOT NULL,
    COST            REAL NOT NULL,
    STATUS          TEXT NOT NULL,
    CREATED_AT      TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

_DDL_AUDIT_LOG = """
CREATE TABLE IF NOT EXISTS AUDIT_LOG (
    ID          INTEGER PRIMARY KEY AUTOINCREMENT,
    NODE        TEXT NOT NULL,
    CUSTOMER_ID TEXT NOT NULL,
    TS          TEXT NOT NULL,
    PAYLOAD     TEXT NOT NULL DEFAULT '{}'
)
"""


def init_graph_tables(conn: sqlite3.Connection) -> None:
    conn.execute(_DDL_CONTACTS)
    conn.execute(_DDL_FULFILLMENTS)
    conn.execute(_DDL_AUDIT_LOG)
    conn.commit()


def record_contact(conn: sqlite3.Connection, customer_id: str,
                   campaign_id: str, contacted_at: str) -> None:
    conn.execute(
        "INSERT INTO GUARDRAIL_CONTACTS (CUSTOMER_ID, CAMPAIGN_ID, CONTACTED_AT) "
        "VALUES (?, ?, ?)",
        (customer_id, campaign_id, contacted_at),
    )
    conn.commit()


def contacts_since(conn: sqlite3.Connection, customer_id: str, since_iso: str) -> int:
    return conn.execute(
        "SELECT count(*) FROM GUARDRAIL_CONTACTS "
        "WHERE CUSTOMER_ID = ? AND CONTACTED_AT >= ?",
        (customer_id, since_iso),
    ).fetchone()[0]


def fulfillment_for(conn: sqlite3.Connection, idempotency_key: str) -> dict | None:
    cur = conn.execute(
        "SELECT * FROM FULFILLMENTS WHERE IDEMPOTENCY_KEY = ?",
        (idempotency_key,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    # row_factory-independent: works with plain-tuple connections too.
    return {d[0]: v for d, v in zip(cur.description, row)}


def insert_fulfillment(conn: sqlite3.Connection, idempotency_key: str,
                       customer_id: str, campaign_id: str, arm: str,
                       cost: float, status: str) -> dict:
    """Insert-or-return. On duplicate IDEMPOTENCY_KEY returns the existing row."""
    existing = fulfillment_for(conn, idempotency_key)
    if existing is not None:
        return existing
    try:
        conn.execute(
            "INSERT INTO FULFILLMENTS "
            "(IDEMPOTENCY_KEY, CUSTOMER_ID, CAMPAIGN_ID, ARM, COST, STATUS) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (idempotency_key, customer_id, campaign_id, arm, cost, status),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.rollback()
        winner = fulfillment_for(conn, idempotency_key)
        if winner is None:
            # Not a duplicate-key race (e.g. NOT NULL violation) — surface it loudly
            # rather than silently breaking the -> dict contract.
            raise
        return winner  # lost the race — return the winning row.
    return fulfillment_for(conn, idempotency_key)
