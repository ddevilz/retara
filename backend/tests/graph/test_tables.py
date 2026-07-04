import sqlite3

import pytest

from magenta.graph.tables import (
    contacts_since,
    fulfillment_for,
    init_graph_tables,
    insert_fulfillment,
    record_contact,
)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    init_graph_tables(c)
    return c


def test_init_is_idempotent(conn):
    init_graph_tables(conn)  # second call must not raise
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {"GUARDRAIL_CONTACTS", "FULFILLMENTS"} <= names


def test_contacts_freq_cap_counter(conn):
    record_contact(conn, "CUST-1", "CAMP-A", "2026-06-20T10:00:00")
    record_contact(conn, "CUST-1", "CAMP-A", "2026-06-25T10:00:00")
    record_contact(conn, "CUST-2", "CAMP-A", "2026-06-25T10:00:00")
    assert contacts_since(conn, "CUST-1", "2026-06-19T00:00:00") == 2
    assert contacts_since(conn, "CUST-1", "2026-06-22T00:00:00") == 1
    assert contacts_since(conn, "CUST-2", "2026-06-19T00:00:00") == 1


def test_fulfillment_insert_and_lookup(conn):
    row = insert_fulfillment(conn, "KEY-1", "CUST-1", "CAMP-A", "BILL_CREDIT", 8.0, "FULFILLED")
    assert row["IDEMPOTENCY_KEY"] == "KEY-1"
    assert row["ARM"] == "BILL_CREDIT"
    got = fulfillment_for(conn, "KEY-1")
    assert got["CUSTOMER_ID"] == "CUST-1"
    assert fulfillment_for(conn, "NOPE") is None


def test_fulfillment_is_idempotent_on_key(conn):
    r1 = insert_fulfillment(conn, "KEY-1", "CUST-1", "CAMP-A", "BILL_CREDIT", 8.0, "FULFILLED")
    r2 = insert_fulfillment(conn, "KEY-1", "CUST-1", "CAMP-A", "DATA_BOOST", 99.0, "FULFILLED")
    # same key => returns original row, no second insert, no overwrite
    assert r2["ARM"] == "BILL_CREDIT"
    n = conn.execute("SELECT count(*) FROM FULFILLMENTS WHERE IDEMPOTENCY_KEY='KEY-1'").fetchone()[0]
    assert n == 1
