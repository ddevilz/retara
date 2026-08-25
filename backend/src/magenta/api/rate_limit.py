"""Shared `Limiter` instance.

Split out from `app.py` so route modules can import `limiter` for the
`@limiter.limit(...)` decorator without a circular import: `app.py` imports
the routers at module load, so a route module importing `limiter` back from
`app.py` would hit a partially-initialized module.
"""
from __future__ import annotations

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from magenta.context import get_tenant


def tenant_rate_key(request: Request) -> str:
    """Rate-limit by tenant, falling back to IP for unauthenticated routes."""
    return get_tenant() or get_remote_address(request)


# ponytail: in-memory limiter state, so limits are per web instance. Correct while
# Phase 1.5's volume constraint keeps the web tier at one instance. Moving to multiple
# instances requires shared storage -- the same decision point as object storage.
limiter = Limiter(key_func=tenant_rate_key, default_limits=["120/minute"])
