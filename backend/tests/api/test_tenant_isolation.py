"""The security proof for Phase 1.2. If any assertion here fails, tenant data is
readable across the boundary and nothing else in this phase matters.
"""
import json

import pytest

from magenta.graph.build import persist_audit
from tests.db_fixtures import TENANT_A


@pytest.mark.asyncio
async def test_audit_rows_do_not_cross_tenants(client, client_tenant_b, db_conn):
    persist_audit(db_conn, TENANT_A, [{
        "NODE": "ACT",
        "CUSTOMER_ID": "CUST_0001",
        "TS": "2026-01-01T00:00:00Z",
        "PAYLOAD": json.dumps({"status": "FULFILLED", "rationale": "TENANT A PRIVATE"}),
    }])

    a = await client.get("/api/audit?customer_id=CUST_0001")
    b = await client_tenant_b.get("/api/audit?customer_id=CUST_0001")

    assert a.status_code == 200 and len(a.json()) == 1
    assert b.status_code == 200 and b.json() == [], "tenant B read tenant A's audit trail"


@pytest.mark.asyncio
async def test_customer_360_audit_is_tenant_scoped(client_tenant_b, db_conn):
    # C0000000 is the first id `generate_population(DEMO_POP_N, DEMO_POP_SEED)`
    # yields (verified against the real demo population -- CUST_0001 does not
    # exist there and would 404, making this assertion vacuous).
    persist_audit(db_conn, TENANT_A, [{
        "NODE": "ACT", "CUSTOMER_ID": "C0000000",
        "TS": "2026-01-01T00:00:00Z", "PAYLOAD": json.dumps({"status": "FULFILLED"}),
    }])
    resp = await client_tenant_b.get("/api/customers/C0000000")
    assert resp.status_code == 200
    assert resp.json()["audit"] == [], "customer 360 leaked another tenant's audit"


@pytest.mark.asyncio
async def test_chat_session_from_other_tenant_is_404(client, client_tenant_b, provisioned_tenants):
    # PRICE_SENSITIVE is not a real archetype (magenta.chat.persona.Archetype
    # has BILL_SHOCK/CONFUSED/PRICE_HAGGLER/NETWORK_COMPLAINER/
    # COMPETITOR_BLUFFER/SLEEPING_DOG) -- verified against the real enum.
    # chat/start now resolves a real per-tenant GraphDeps (Task 5), which needs
    # trained models on disk -- provisioned_tenants trains and saves them for
    # both TENANT_A and TENANT_B under an isolated MAGENTA_MODEL_DIR.
    start = await client.post(
        "/api/chat/start", json={"mode": "persona", "archetype": "PRICE_HAGGLER"}
    )
    assert start.status_code == 200
    session_id = start.json()["session_id"]

    resp = await client_tenant_b.post(f"/api/chat/{session_id}/turn", json={"text": "hi"})
    assert resp.status_code == 404, "tenant B reached tenant A's chat session"
