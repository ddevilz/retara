from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_health_does_not_touch_the_database(unauthenticated_client):
    """Liveness must not depend on Postgres, or a DB blip becomes a restart loop."""
    with patch("magenta.api.app.get_conn", side_effect=RuntimeError("db down")):
        resp = await unauthenticated_client.get("/api/health")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_ready_returns_200_when_db_is_up(unauthenticated_client, migrated_db):
    resp = await unauthenticated_client.get("/api/ready")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_ready_returns_503_when_db_is_down(unauthenticated_client):
    with patch("magenta.api.app.get_conn", side_effect=RuntimeError("db down")):
        resp = await unauthenticated_client.get("/api/ready")
    assert resp.status_code == 503


def test_cors_origins_come_from_env(monkeypatch):
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://a.example,https://b.example")
    from magenta.api.app import allowed_origins

    assert allowed_origins() == ["https://a.example", "https://b.example"]


def test_cors_defaults_to_local_dev(monkeypatch):
    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
    from magenta.api.app import allowed_origins

    assert "http://localhost:5173" in allowed_origins()
