"""verify_token is the single seam we mock, mirroring how magenta.llm.chat is mocked.
No test here reaches Clerk."""
from unittest.mock import ANY, patch

import pytest
from fastapi import HTTPException

from magenta.auth import (
    AuthError,
    ClerkClaims,
    TenantContext,
    current_tenant,
    verify_token,
)


def test_claims_parse_with_org():
    c = ClerkClaims(user_id="user_1", org_id="org_1", org_role="org:admin", session_id="sess_1")
    assert c.org_id == "org_1"


def test_claims_allow_missing_org():
    """A Clerk user with no ACTIVE organization gets a token with no org_id.
    That must parse cleanly here and be rejected later by the dependency, not crash."""
    c = ClerkClaims(user_id="user_1", org_id=None, org_role=None, session_id="sess_1")
    assert c.org_id is None


def test_verify_token_rejects_empty():
    with pytest.raises(AuthError):
        verify_token("")


def test_verify_token_wraps_sdk_failure():
    with patch("magenta.auth._sdk_verify", side_effect=ValueError("bad signature")):
        with pytest.raises(AuthError):
            verify_token("some.jwt.token")


def test_verify_token_rejects_malformed_payload():
    """A payload missing 'sub' must surface as AuthError, not a bare KeyError —
    this is the security boundary; nothing downstream should see a raw SDK/parsing
    exception."""
    with patch("magenta.auth._sdk_verify", return_value={}):
        with pytest.raises(AuthError):
            verify_token("some.jwt.token")


def test_tenant_context_shape():
    t = TenantContext(tenant_id="org_1", user_id="user_1", role="org:admin")
    assert t.tenant_id == "org_1"


def test_missing_header_is_401():
    with pytest.raises(HTTPException) as exc:
        current_tenant(authorization=None)
    assert exc.value.status_code == 401


def test_non_bearer_header_is_401():
    with pytest.raises(HTTPException) as exc:
        current_tenant(authorization="Basic abc123")
    assert exc.value.status_code == 401


def test_invalid_token_is_401():
    with patch("magenta.auth.verify_token", side_effect=AuthError("nope")):
        with pytest.raises(HTTPException) as exc:
            current_tenant(authorization="Bearer bad.token")
    assert exc.value.status_code == 401


def test_authenticated_without_org_is_403():
    """A real Clerk user who has not selected an active organization. Authenticated, but
    there is no tenant to act as -- 403, not 401, and never a 500."""
    claims = ClerkClaims(user_id="user_1", org_id=None, org_role=None, session_id="s")
    with patch("magenta.auth.verify_token", return_value=claims):
        with pytest.raises(HTTPException) as exc:
            current_tenant(authorization="Bearer good.token")
    assert exc.value.status_code == 403


def test_valid_org_token_yields_tenant_context():
    claims = ClerkClaims(
        user_id="user_1", org_id="org_xyz", org_role="org:admin", session_id="s"
    )
    with patch("magenta.auth.verify_token", return_value=claims), \
         patch("magenta.auth.ensure_org") as mock_ensure_org:
        ctx = current_tenant(authorization="Bearer good.token")
    mock_ensure_org.assert_called_once_with(ANY, "org_xyz", "org_xyz")
    assert ctx.tenant_id == "org_xyz"
    assert ctx.user_id == "user_1"
    assert ctx.role == "org:admin"
