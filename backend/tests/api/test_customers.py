import pytest

from magenta.api import data_access as da


@pytest.fixture(autouse=True)
def no_real_db(tmp_path, monkeypatch):
    """Point audit reads at a DB path that never exists.

    data/magenta.db is a live app artifact (a background ablation run may be
    writing it right now) — tests must never open it. Pointing DB_PATH at a
    nonexistent tmp path makes data_access._open_db() return None, so
    audit_rows() deterministically returns [] without touching real state.
    """
    monkeypatch.setattr(da, "DB_PATH", tmp_path / "no-such-magenta.db")


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
