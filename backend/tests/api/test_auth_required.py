"""Every data route rejects an unauthenticated request. /api/health does not.

`test_route_requires_auth` used to walk a hand-maintained ROUTES list -- its own
docstring admitted a new unprotected route only fails here if someone remembers to
add it to the list. Replaced with a structural invariant: walk every route FastAPI
actually registered and assert `current_tenant` is one of its dependencies. A new
route added without `Depends(current_tenant)` now fails this test with no list to
remember to update.

FastAPI 0.139 groups included routers behind `_IncludedRouter` wrappers in
`app.routes` rather than flattening their routes in-place (verified against the
installed version -- `app.routes` alone only shows /api/health plus the doc
routes); the real APIRoute objects live at `router.original_router.routes`.
"""
import pytest

from magenta.api.app import create_app
from magenta.auth import current_tenant


def _iter_api_routes(routes):
    """Yield every APIRoute, whether registered directly or nested inside an
    `_IncludedRouter` (any router mounted via `include_router`)."""
    for route in routes:
        path = getattr(route, "path", None)
        if path is not None:
            yield route
            continue
        original_router = getattr(route, "original_router", None)
        if original_router is not None:
            yield from _iter_api_routes(original_router.routes)


def test_every_api_route_requires_a_tenant():
    """A new unprotected route must fail here without anyone remembering to add it."""
    app = create_app()
    unprotected = []
    for route in _iter_api_routes(app.routes):
        path = getattr(route, "path", "")
        if not path.startswith("/api") or path == "/api/health":
            continue
        dependant = getattr(route, "dependant", None)
        deps = [d.call for d in dependant.dependencies] if dependant else []
        if current_tenant not in deps:
            unprotected.append(f"{getattr(route, 'methods', '')} {path}")
    assert unprotected == [], f"unprotected /api routes: {unprotected}"


@pytest.mark.asyncio
async def test_health_is_public(unauthenticated_client):
    resp = await unauthenticated_client.get("/api/health")
    assert resp.status_code == 200
