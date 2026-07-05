"""Shared fixtures for API tests. LLM is always mocked here; nothing hits network."""
from __future__ import annotations

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from magenta.api.app import create_app


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
