"""Clerk session verification and tenant resolution.

`verify_token` is a deliberately thin wrapper around the Clerk SDK so tests have ONE
seam to mock — the same convention this repo already uses for `magenta.llm.chat`.
We do not re-implement JWT verification: signature checking, key rotation, and clock
skew are the SDK's job and are a bad place to be clever.

`CLERK_AUTHORIZED_PARTIES` is not optional. Without an `azp` allowlist, a token minted
for a different application on the same Clerk instance verifies successfully here.

NOTE on SDK import path: the brief this module was drafted from assumed
`clerk_backend_api.jwks_helpers.verify_token` / `AuthenticateRequestOptions`. Neither
exists in the installed `clerk-backend-api==7.0.0` — that module does not exist at
all. The real, networkless, single-token verifier lives at
`clerk_backend_api.security.verify_token`, configured with
`clerk_backend_api.security.VerifyTokenOptions` (a dataclass, not
`AuthenticateRequestOptions` — that class configures `authenticate_request`, a
different higher-level flow that verifies a whole inbound HTTP request, not a bare
token string). Verified via `pkgutil.iter_modules` + grepping the installed package;
see task-1-report.md.
"""
from __future__ import annotations

import os

from clerk_backend_api.security import VerifyTokenOptions
from clerk_backend_api.security import verify_token as clerk_verify
from fastapi import Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.engine import Connection

from magenta.db import get_conn
from magenta.jobs import train_tenant_models_job
from magenta.logging_config import bind_tenant, get_logger

logger = get_logger(__name__)


class AuthError(Exception):
    """Any failure to establish a verified identity. Never leaks SDK internals."""


class ClerkClaims(BaseModel):
    user_id: str
    org_id: str | None = None
    org_role: str | None = None
    session_id: str


class TenantContext(BaseModel):
    """A resolved, authorised tenant. Every data route depends on one of these."""
    tenant_id: str
    user_id: str
    role: str


def _secret_key() -> str:
    key = os.environ.get("CLERK_SECRET_KEY")
    if not key:
        raise RuntimeError("CLERK_SECRET_KEY is not set")
    return key


def _authorized_parties() -> list[str]:
    raw = os.environ.get("CLERK_AUTHORIZED_PARTIES", "")
    parties = [p.strip() for p in raw.split(",") if p.strip()]
    if not parties:
        raise RuntimeError(
            "CLERK_AUTHORIZED_PARTIES is not set. Without an azp allowlist, a token "
            "issued for another app on this Clerk instance would be accepted."
        )
    return parties


def _sdk_verify(token: str) -> dict:
    """Verification against the instance's signing key. Networkless once that key is
    cached locally; on a cache miss (e.g. key rotation, unseen `kid`) the SDK fetches
    JWKS from Clerk over HTTPS and caches the PEM by `kid`. Isolated as its own
    function so tests patch exactly this and nothing else."""
    return clerk_verify(
        token,
        VerifyTokenOptions(
            secret_key=_secret_key(),
            authorized_parties=_authorized_parties(),
        ),
    )


def verify_token(token: str) -> ClerkClaims:
    if not token:
        raise AuthError("missing token")
    try:
        payload = _sdk_verify(token)
    except RuntimeError:
        # Missing CLERK_SECRET_KEY / CLERK_AUTHORIZED_PARTIES: a misconfigured
        # deploy, not an invalid token. Must NOT be caught by the generic
        # except below -- that would surface as a silent 401 on every request
        # with no log line anywhere. Let it propagate to a 500 instead.
        raise
    except Exception as exc:  # SDK/payload failures alike mean "not authenticated"
        raise AuthError("invalid token") from exc
    try:
        return ClerkClaims(
            user_id=payload["sub"],
            org_id=payload.get("org_id"),
            org_role=payload.get("org_role"),
            session_id=payload.get("sid", ""),
        )
    except Exception as exc:  # malformed payload (missing 'sub', etc.)
        raise AuthError("invalid token") from exc


def ensure_org(conn: Connection, org_id: str, name: str) -> None:
    """Get-or-create the tenant, and on creation enqueue provisioning in the SAME
    transaction.

    This replaces a Clerk webhook sync: the first time anyone from an organization
    calls the API, the tenant exists. No webhook endpoint, no reconciliation job,
    no drift between Clerk and our registry.

    `rowcount == 1` means this call created the row, so exactly one provisioning job
    is enqueued per tenant no matter how many requests arrive. `driver_connection`
    reaches the raw psycopg connection underneath SQLAlchemy, which is what
    Procrastinate needs in order to write the job through our transaction rather
    than its own. `queueing_lock=org_id` scopes the dedupe lock per tenant instead
    of the task's global default -- otherwise two tenants signing up around the same
    time would have the second `.defer()` raise `AlreadyEnqueued`. `lock=org_id` is
    the separate primitive that actually matters for correctness once a job is
    running: `queueing_lock` only dedupes jobs still sitting in `todo`, it does NOT
    stop two jobs with the same lock value from *executing* concurrently. Without
    `lock`, a stalled-job retry (see `jobs.retry_stalled_jobs_job`) that re-queues a
    tenant whose worker only *looked* dead (heartbeat lapsed mid multi-minute
    LightGBM fit) could have two workers training and saving the same tenant's
    artifacts at once -- `RiskModel.save()` is a bare `joblib.dump`, not a
    tmp-file-then-rename, so that race can torn-write the file.

    No commit here: the caller owns the transaction boundary. That is the entire
    point -- if the caller rolls back, the job disappears with the row.
    """
    result = conn.execute(
        text(
            'INSERT INTO "ORGANIZATIONS" ("ID", "NAME") VALUES (:id, :name) '
            'ON CONFLICT ("ID") DO NOTHING'
        ),
        {"id": org_id, "name": name},
    )
    if result.rowcount == 1:
        train_tenant_models_job.configure(
            connection=conn.connection.driver_connection,
            queueing_lock=org_id,
            lock=org_id,
        ).defer(tenant_id=org_id)


def current_tenant(
    authorization: str | None = Header(None),
) -> TenantContext:
    """Sync on purpose: this does a blocking psycopg INSERT via `ensure_org`, then
    commits here (the commit moved out of `ensure_org` so its own transaction-boundary
    tests can control commit/rollback), and `verify_token` can do a blocking HTTPS
    JWKS fetch on a `kid` cache miss.
    An `async def` here would run that I/O directly on the event loop and stall
    every in-flight SSE stream; FastAPI runs a sync dependency in the threadpool
    automatically."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()

    try:
        claims = verify_token(token)
    except AuthError:
        raise HTTPException(status_code=401, detail="invalid token") from None

    if not claims.org_id:
        raise HTTPException(
            status_code=403,
            detail="no active organization; select one to continue",
        )

    # ponytail: unconditional upsert on every authenticated request. Negligible at
    # current scale; if it shows up in a profile, skip the write once the org is
    # known-present (a bounded TTL cache, as Phase 1.3 uses for deps).
    with get_conn() as conn:
        ensure_org(conn, claims.org_id, claims.org_id)
        conn.commit()

    return TenantContext(
        tenant_id=claims.org_id,
        user_id=claims.user_id,
        role=claims.org_role or "org:member",
    )


async def bound_tenant(
    tenant: TenantContext = Depends(current_tenant),
) -> TenantContext:
    """Routes depend on this instead of `current_tenant` directly.

    `current_tenant` is a sync `def`, so FastAPI runs it in a worker thread via
    `anyio.to_thread.run_sync`; a `bind_contextvars` call made inside it would bind
    into that thread's OWN COPY of the context and be discarded the instant the
    thread returns -- the route handler and `RequestContextMiddleware` never see it.
    This wrapper is `async def`, so FastAPI resolves it directly on the event loop,
    in the same context the handler and middleware share -- binding here actually
    survives into the rest of the request.
    """
    bind_tenant(tenant.tenant_id)
    logger.info("tenant.resolved", tenant_id=tenant.tenant_id)
    return tenant
