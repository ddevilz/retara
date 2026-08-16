"""The cross-tenant collision regression. Before the fix, `idempotency_key` hashed
only customer:campaign:arm, and FULFILLMENTS had a global PRIMARY KEY — so tenant B's
genuine offer was silently swallowed as a duplicate of tenant A's."""
from datetime import datetime, timedelta, timezone

import pytest

from magenta.graph.tables import (
    contacts_since,
    fulfillment_for,
    idempotency_key,
    insert_fulfillment,
    record_contact,
)
from magenta.offers import Arm
from tests.db_fixtures import TENANT_A, TENANT_B


def test_idempotency_key_differs_across_tenants():
    a = idempotency_key(TENANT_A, "CUST_0001", "CAMP-1", Arm.BILL_CREDIT)
    b = idempotency_key(TENANT_B, "CUST_0001", "CAMP-1", Arm.BILL_CREDIT)
    assert a != b


def test_two_tenants_same_customer_id_both_fulfill(db_conn):
    """The bug: identical customer IDs across tenants must not collide."""
    key_a = idempotency_key(TENANT_A, "CUST_0001", "CAMP-1", Arm.BILL_CREDIT)
    key_b = idempotency_key(TENANT_B, "CUST_0001", "CAMP-1", Arm.BILL_CREDIT)

    row_a = insert_fulfillment(
        db_conn, TENANT_A, key_a, "CUST_0001", "CAMP-1", "BILL_CREDIT", 8.0, "FULFILLED"
    )
    row_b = insert_fulfillment(
        db_conn, TENANT_B, key_b, "CUST_0001", "CAMP-1", "BILL_CREDIT", 8.0, "FULFILLED"
    )

    assert row_a["IDEMPOTENCY_KEY"] != row_b["IDEMPOTENCY_KEY"]
    assert fulfillment_for(db_conn, TENANT_A, key_b) is None, "tenant A must not see B's row"
    assert fulfillment_for(db_conn, TENANT_B, key_a) is None, "tenant B must not see A's row"


def test_insert_fulfillment_is_idempotent_within_a_tenant(db_conn):
    key = idempotency_key(TENANT_A, "CUST_0002", "CAMP-1", Arm.DATA_BOOST)
    first = insert_fulfillment(
        db_conn, TENANT_A, key, "CUST_0002", "CAMP-1", "DATA_BOOST", 5.0, "FULFILLED"
    )
    second = insert_fulfillment(
        db_conn, TENANT_A, key, "CUST_0002", "CAMP-1", "DATA_BOOST", 999.0, "FULFILLED"
    )
    assert second["COST"] == first["COST"] == 5.0, "replay must return the winning row, not overwrite"


def test_contacts_since_is_tenant_scoped(db_conn):
    now = datetime.now(timezone.utc)
    record_contact(db_conn, TENANT_A, "CUST_0003", "CAMP-1", now)
    record_contact(db_conn, TENANT_B, "CUST_0003", "CAMP-1", now)
    since = now - timedelta(days=1)
    assert contacts_since(db_conn, TENANT_A, "CUST_0003", since) == 1
