"""The single Postgres fixture for the whole suite.

Replaces ~20 hand-rolled `sqlite3.connect(":memory:")` sites. Each test runs inside
a transaction that is rolled back on teardown, so tests stay isolated without paying
to re-run migrations per test.
"""
from __future__ import annotations

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.engine import Connection

from magenta.config import repo_root
from magenta.db import get_engine

TENANT_A = "org_test_aaa"
TENANT_B = "org_test_bbb"


@pytest.fixture(scope="session")
def migrated_db() -> None:
    """Run migrations once per session against DATABASE_URL."""
    cfg = Config(str(repo_root() / "backend" / "alembic.ini"))
    cfg.set_main_option("script_location", str(repo_root() / "backend" / "alembic"))
    # No downgrade-to-base here: upgrade(head) is idempotent, and DATABASE_URL may
    # point at a developer's working database. Per-test isolation comes from the
    # transaction rollback in `db_conn`, not from dropping the schema.
    command.upgrade(cfg, "head")


@pytest.fixture
def db_conn(migrated_db) -> Connection:
    """A connection inside a transaction that is always rolled back.

    Code under test calls `conn.commit()`; SQLAlchemy nests that inside this outer
    transaction, so the rollback here still discards it.
    """
    engine = get_engine()
    conn = engine.connect()
    trans = conn.begin()
    try:
        yield conn
    finally:
        trans.rollback()
        conn.close()
