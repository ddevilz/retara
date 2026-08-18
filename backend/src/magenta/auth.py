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

from clerk_backend_api.security import VerifyTokenOptions, verify_token as clerk_verify
from pydantic import BaseModel


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
        return ClerkClaims(
            user_id=payload["sub"],
            org_id=payload.get("org_id"),
            org_role=payload.get("org_role"),
            session_id=payload.get("sid", ""),
        )
    except Exception as exc:  # SDK/payload failures alike mean "not authenticated"
        raise AuthError("invalid token") from exc
