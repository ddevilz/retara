import pytest
from fastapi.testclient import TestClient

from magenta.api.app import create_app


@pytest.mark.asyncio
async def test_health_ok(client):
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "magenta-retain"


def test_lifespan_opens_and_closes_procrastinate_app():
    """The `client` fixture above uses ASGITransport directly, which never triggers
    ASGI lifespan events -- so it can't prove `procrastinate_app.open()`/`.close()`
    (registered in `_lifespan`, magenta/api/app.py) actually run without raising.
    TestClient drives the real lifespan protocol, so this is the one test in the
    suite that exercises it."""
    with TestClient(create_app()) as client:
        resp = client.get("/api/health")
        assert resp.status_code == 200
