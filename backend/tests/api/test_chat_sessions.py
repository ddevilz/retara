"""A leaked or guessed session id must not cross a tenant boundary."""
from magenta.api import chat_sessions
from magenta.api.chat_sessions import ChatSession
from tests.db_fixtures import TENANT_A, TENANT_B


def test_get_rejects_other_tenants_session():
    chat_sessions.clear()
    sid = chat_sessions.new_id()
    chat_sessions.create(
        ChatSession(
            session_id=sid, tenant_id=TENANT_A, mode="persona",
            customer_id="CUST_0001", archetype="PRICE_SENSITIVE", chat=object(),
        )
    )
    assert chat_sessions.get(sid, TENANT_A) is not None
    assert chat_sessions.get(sid, TENANT_B) is None


def test_get_unknown_session_is_none():
    chat_sessions.clear()
    assert chat_sessions.get("sess-doesnotexist", TENANT_A) is None
