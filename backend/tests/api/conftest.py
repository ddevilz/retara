"""LLM and auth are both mocked here; nothing hits network."""
from __future__ import annotations

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from magenta.api.app import create_app
from magenta.auth import TenantContext, current_tenant
from tests.db_fixtures import TENANT_A, TENANT_B


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    """Authenticated as TENANT_A. The override replaces the whole dependency, so no
    test needs a real Clerk token."""
    app = create_app()
    app.dependency_overrides[current_tenant] = lambda: TenantContext(
        tenant_id=TENANT_A, user_id="user_test", role="org:admin"
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def unauthenticated_client() -> AsyncClient:
    """No override — exercises the real dependency and therefore the real 401 path."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def client_tenant_b() -> AsyncClient:
    app = create_app()
    app.dependency_overrides[current_tenant] = lambda: TenantContext(
        tenant_id=TENANT_B, user_id="user_other", role="org:admin"
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
