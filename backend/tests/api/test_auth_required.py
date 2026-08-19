"""Every data route rejects an unauthenticated request. /api/health does not.

Parametrised deliberately: a new route added without auth fails here only if it is
added to this list, so the list is also the checklist during review.
"""
import pytest

ROUTES = [
    ("GET", "/api/scorecards", None),
    ("GET", "/api/customers", None),
    ("GET", "/api/customers/CUST_0001", None),
    ("GET", "/api/audit?customer_id=CUST_0001", None),
    ("POST", "/api/run-one", {"customer_id": "CUST_0001"}),
    ("POST", "/api/experiment", {"policy": "agent", "n": 2, "seed": 7}),
    ("POST", "/api/chat/start", {"mode": "persona", "archetype": "PRICE_SENSITIVE"}),
    ("POST", "/api/chat/sess-abc/turn", {"text": "hello"}),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("method,path,body", ROUTES)
async def test_route_requires_auth(unauthenticated_client, method, path, body):
    resp = await unauthenticated_client.request(method, path, json=body)
    assert resp.status_code == 401, f"{method} {path} is not protected"


@pytest.mark.asyncio
async def test_health_is_public(unauthenticated_client):
    resp = await unauthenticated_client.get("/api/health")
    assert resp.status_code == 200
