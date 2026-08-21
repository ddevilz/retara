from magenta.budget import record_usage, tokens_used_this_month
from tests.db_fixtures import TENANT_A, TENANT_B


def test_usage_accumulates_per_tenant(db_conn):
    record_usage(TENANT_A, "cheap", "llama-3.1-8b", 100, 50)
    record_usage(TENANT_A, "large", "llama-3.3-70b", 200, 100)
    assert tokens_used_this_month(TENANT_A) == 450


def test_usage_is_tenant_isolated(db_conn):
    record_usage(TENANT_A, "cheap", "llama-3.1-8b", 100, 50)
    assert tokens_used_this_month(TENANT_B) == 0


def test_unknown_tenant_has_zero_usage(db_conn):
    assert tokens_used_this_month("org_never_seen") == 0
