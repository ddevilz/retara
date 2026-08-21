import pytest


@pytest.mark.asyncio
async def test_unprovisioned_tenant_gets_503(client, monkeypatch, tmp_path):
    """503 not 500: the tenant is valid, the models simply are not ready yet."""
    monkeypatch.setenv("MAGENTA_MODEL_DIR", str(tmp_path))
    from magenta.api.deps import DEPS_CACHE

    DEPS_CACHE.clear()
    resp = await client.post("/api/run-one", json={"customer_id": "CUST_0001"})
    assert resp.status_code == 503
    assert "Retry-After" in resp.headers
