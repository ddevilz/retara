"""DATABASE_URL-driven engine. No :memory: SQLite anywhere in this repo's tests."""
import pytest
from sqlalchemy import text

from magenta.db import database_url, get_conn, get_engine


def test_database_url_requires_env(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    get_engine.cache_clear()
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        database_url()


def test_database_url_normalizes_bare_postgresql_scheme(monkeypatch):
    """Render's `fromDatabase.property: connectionString` yields a bare
    `postgresql://...` URL. SQLAlchemy defaults that to the psycopg2 dialect,
    which this repo does not install (only `psycopg[binary]`) -- first DB
    access would raise ImportError on a real deploy. `database_url()` must
    rewrite the scheme to `postgresql+psycopg://`."""
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://magenta:magenta@localhost:5433/magenta"
    )
    assert database_url() == "postgresql+psycopg://magenta:magenta@localhost:5433/magenta"


def test_database_url_leaves_psycopg_scheme_unchanged(monkeypatch):
    """Already-correct URLs (local dev, and this normalization applied once)
    must pass through byte-for-byte -- no double-rewrite."""
    url = "postgresql+psycopg://magenta:magenta@localhost:5433/magenta"
    monkeypatch.setenv("DATABASE_URL", url)
    assert database_url() == url


def test_get_conn_executes_against_postgres():
    with get_conn() as conn:
        assert conn.execute(text("SELECT 1")).scalar() == 1


def test_engine_is_postgres_not_sqlite():
    assert get_engine().dialect.name == "postgresql"


def test_committed_write_does_not_leak_out_of_a_test(db_conn):
    """A commit INSIDE a test must not survive teardown. This is the regression guard
    for the fixture that previously claimed rollback isolation and did not have it."""
    db_conn.execute(text(
        'INSERT INTO "GUARDRAIL_CONTACTS" ("TENANT_ID", "CUSTOMER_ID", "CAMPAIGN_ID", "CONTACTED_AT") '
        "VALUES (:t, :c, :k, now())"
    ), {"t": "org_leak_probe", "c": "CUST-LEAK", "k": "CAMP-LEAK"})
    db_conn.commit()   # the exact thing the old fixture could not undo
    assert db_conn.execute(text(
        'SELECT count(*) FROM "GUARDRAIL_CONTACTS" WHERE "TENANT_ID" = :t'
    ), {"t": "org_leak_probe"}).scalar_one() == 1


def test_previous_tests_committed_rows_are_gone(db_conn):
    assert db_conn.execute(text(
        'SELECT count(*) FROM "GUARDRAIL_CONTACTS" WHERE "TENANT_ID" = :t'
    ), {"t": "org_leak_probe"}).scalar_one() == 0
