import json

import pytest


def test_audit_rows_are_tenant_scoped(db_conn):
    from magenta.api.data_access import audit_rows
    from magenta.graph.build import persist_audit
    from tests.db_fixtures import TENANT_A, TENANT_B

    # persist_audit's entry shape is graph.nodes._audit()'s output: uppercase
    # keys, PAYLOAD already json.dumps()d (see graph/nodes.py::_audit).
    entry = [{"NODE": "ACT", "CUSTOMER_ID": "CUST_0001",
              "TS": "2026-01-01T00:00:00+00:00",
              "PAYLOAD": json.dumps({"status": "FULFILLED"})}]
    persist_audit(db_conn, TENANT_A, entry)

    assert len(audit_rows(TENANT_A, "CUST_0001")) == 1
    assert audit_rows(TENANT_B, "CUST_0001") == []


def test_open_db_is_gone():
    """One connection path only."""
    import magenta.api.data_access as da
    assert not hasattr(da, "_open_db")
    assert not hasattr(da, "DB_PATH")


@pytest.mark.asyncio
async def test_customers_limit(client):
    resp = await client.get("/api/customers?limit=5")
    assert resp.status_code == 200
    rows = resp.json()
    assert 0 < len(rows) <= 5
    # Observable-only: hidden simulator fields must NEVER be present.
    leak = {"theta_churn", "theta_price_sens", "persuadable_segment",
            "competitor_pull", "theta_churn_base"}
    assert not (set(rows[0].keys()) & leak), "hidden state leaked into API"


@pytest.mark.asyncio
async def test_customer_360_and_404(client):
    rows = (await client.get("/api/customers?limit=1")).json()
    cid = rows[0]["customer_id"]
    ok = await client.get(f"/api/customers/{cid}")
    assert ok.status_code == 200
    body = ok.json()
    assert body["customer"]["customer_id"] == cid
    assert isinstance(body["audit"], list)   # may be empty if graph not run

    missing = await client.get("/api/customers/NOPE-99999")
    assert missing.status_code == 404
