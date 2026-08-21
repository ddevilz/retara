"""The cross-tenant collision regression. Before the fix, `idempotency_key` hashed
only customer:campaign:arm, and FULFILLMENTS had a global PRIMARY KEY — so tenant B's
genuine offer was silently swallowed as a duplicate of tenant A's."""
from datetime import UTC, datetime, timedelta

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


def test_same_literal_key_isolated_by_composite_primary_key(db_conn):
    """The actual invariant this branch is named for: FULFILLMENTS' primary
    key is (TENANT_ID, IDEMPOTENCY_KEY), not IDEMPOTENCY_KEY alone. Insert the
    SAME literal key under two tenants and both rows must persist
    independently -- proving isolation without leaning on idempotency_key's
    own tenant-hashing to manufacture different keys."""
    same_key = "not-a-hash-just-a-literal-key"
    insert_fulfillment(
        db_conn, TENANT_A, same_key, "CUST_0001", "CAMP-1", "BILL_CREDIT", 8.0, "FULFILLED"
    )
    insert_fulfillment(
        db_conn, TENANT_B, same_key, "CUST_0001", "CAMP-1", "BILL_CREDIT", 8.0, "FULFILLED"
    )
    assert fulfillment_for(db_conn, TENANT_A, same_key) is not None
    assert fulfillment_for(db_conn, TENANT_B, same_key) is not None


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
    now = datetime.now(UTC)
    record_contact(db_conn, TENANT_A, "CUST_0003", "CAMP-1", now)
    record_contact(db_conn, TENANT_B, "CUST_0003", "CAMP-1", now)
    since = now - timedelta(days=1)
    assert contacts_since(db_conn, TENANT_A, "CUST_0003", since) == 1


def test_contacts_since_filters_by_window_boundary(db_conn):
    """Boundary coverage: contacts_since backs the contact frequency cap, so the
    `>= :since` clause has to actually filter, not just exist. Without this,
    deleting the clause leaves the suite green (it did -- see final-fix-report)."""
    now = datetime.now(UTC)
    since = now - timedelta(days=14)
    record_contact(db_conn, TENANT_A, "CUST_0004", "CAMP-1", now - timedelta(days=7))   # inside
    record_contact(db_conn, TENANT_A, "CUST_0004", "CAMP-1", now - timedelta(days=20))  # outside
    assert contacts_since(db_conn, TENANT_A, "CUST_0004", since) == 1
