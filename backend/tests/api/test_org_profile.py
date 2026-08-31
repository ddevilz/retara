"""GET/PUT /api/org/profile: tenant isolation, admin gating, industry validation."""
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from magenta.api.app import create_app
from magenta.auth import TenantContext, current_tenant
from tests.db_fixtures import TENANT_A


@pytest.mark.asyncio
async def test_get_profile_returns_current_values(client, db_conn):
    db_conn.execute(
        text('UPDATE "ORGANIZATIONS" SET "NAME" = :n WHERE "ID" = :id'),
        {"n": "Acme Telecom", "id": TENANT_A},
    )
    db_conn.commit()
    resp = await client.get("/api/org/profile")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Acme Telecom"
    assert body["industry"] is None


@pytest.mark.asyncio
async def test_put_profile_updates_and_returns(client, db_conn):
    resp = await client.put(
        "/api/org/profile",
        json={
            "name": "Acme Telecom",
            "industry": "telecom",
            "monthly_token_budget": 100000,
            "admin_contact_email": "ops@acme.example",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["industry"] == "telecom"
    assert body["monthly_token_budget"] == 100000
    assert body["admin_contact_email"] == "ops@acme.example"


@pytest.mark.asyncio
async def test_put_profile_rejects_unsupported_industry(client, db_conn):
    resp = await client.put(
        "/api/org/profile",
        json={"name": "Acme", "industry": "healthcare"},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_put_profile_requires_admin_role(db_conn):
    """The `client` fixture defaults to org:admin -- build a member-role client
    inline rather than adding a new global fixture for one test."""
    app = create_app()
    app.dependency_overrides[current_tenant] = lambda: TenantContext(
        tenant_id=TENANT_A, user_id="user_test", role="org:member"
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.put(
            "/api/org/profile", json={"name": "Acme", "industry": "telecom"}
        )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_profile_is_tenant_isolated(client, client_tenant_b, db_conn):
    await client.put(
        "/api/org/profile", json={"name": "Tenant A Co", "industry": "telecom"}
    )
    resp_b = await client_tenant_b.get("/api/org/profile")
    assert resp_b.json()["name"] != "Tenant A Co"
