"""Item 1 regression: /api/run-one must run the graph under the REQUEST's
tenant, not the cached GraphDeps singleton's DEFAULT_TENANT_ID.

Pre-fix, two tenants running /api/run-one for the same customer_id collide on
idempotency_key(DEFAULT_TENANT_ID, ...) -- the second tenant's real offer is
suppressed as a duplicate and the guardrail frequency cap is shared across
tenants. This drives the REAL graph (build_graph + nodes.act) against Postgres
via two authenticated clients and asserts two independent FULFILLMENTS rows.
"""
from __future__ import annotations

import pytest
from sqlalchemy import text

import magenta.graph.nodes as nodes_mod
from magenta.api import routes_stream as rs
from magenta.graph.build import GraphDeps
from tests.db_fixtures import TENANT_A, TENANT_B
from tests.graph.conftest import (
    FakeBandit,
    FakeCatalog,
    FakeCustomer,
    FakeOracle,
    FakeRisk,
    FakeUplift,
    SpyChat,
)


class _Params:
    freq_cap_days = 14
    freq_cap_max = 1
    value_cap = 40.0
    p90_clv = 2000.0


@pytest.fixture
def real_graph_deps(monkeypatch, db_conn):
    """A real GraphDeps (fakes for ML/oracle, real Postgres conn) shared as the
    cached singleton `get_graph_deps()` returns -- tenant_id here is the
    process-default and must be overridden per-request by the route, exactly
    like the production `get_graph_deps()` this replaces."""
    customer = FakeCustomer(customer_id="CUST-TENANT-SCOPE")
    deps = GraphDeps(
        risk=FakeRisk(), uplift=FakeUplift(), bandit=FakeBandit(),
        catalog=FakeCatalog(), oracle=FakeOracle(), conn=db_conn,
        params=_Params(), chat=SpyChat(), load_customer=lambda cid: customer,
        checkpointer=None,
    )
    monkeypatch.setattr(nodes_mod, "featurize", lambda c: [0.0])
    monkeypatch.setattr(rs, "get_graph_deps", lambda: deps)
    monkeypatch.setattr(rs, "_find_customer", lambda cid: customer)
    return deps


@pytest.mark.asyncio
async def test_run_one_scopes_fulfillment_by_request_tenant(
    client, client_tenant_b, real_graph_deps, db_conn
):
    resp_a = await client.post("/api/run-one", json={"customer_id": "CUST-TENANT-SCOPE"})
    resp_b = await client_tenant_b.post("/api/run-one", json={"customer_id": "CUST-TENANT-SCOPE"})
    assert resp_a.status_code == 200 and resp_b.status_code == 200

    rows = db_conn.execute(
        text('SELECT "TENANT_ID", "STATUS" FROM "FULFILLMENTS" WHERE "CUSTOMER_ID" = :c'),
        {"c": "CUST-TENANT-SCOPE"},
    ).mappings().all()
    tenant_ids = {r["TENANT_ID"] for r in rows}
    assert len(rows) == 2, f"expected one FULFILLMENTS row per tenant, got {rows}"
    assert tenant_ids == {TENANT_A, TENANT_B}
    assert all(r["STATUS"] == "FULFILLED" for r in rows)

    # Neither tenant's offer was treated as an idempotent replay of the other's.
    contacts = db_conn.execute(
        text('SELECT "TENANT_ID" FROM "GUARDRAIL_CONTACTS" WHERE "CUSTOMER_ID" = :c'),
        {"c": "CUST-TENANT-SCOPE"},
    ).mappings().all()
    assert {r["TENANT_ID"] for r in contacts} == {TENANT_A, TENANT_B}
    for text_body, label in ((resp_a.text, "A"), (resp_b.text, "B")):
        assert "IDEMPOTENT_HIT" not in text_body, f"tenant {label} was treated as a duplicate"
