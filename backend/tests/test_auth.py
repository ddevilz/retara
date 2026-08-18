"""verify_token is the single seam we mock, mirroring how magenta.llm.chat is mocked.
No test here reaches Clerk."""
from unittest.mock import patch

import pytest

from magenta.auth import AuthError, ClerkClaims, TenantContext, verify_token


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
