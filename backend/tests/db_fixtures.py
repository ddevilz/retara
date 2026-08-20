"""The single Postgres fixture for the whole suite.

Replaces ~20 hand-rolled in-memory-db connection sites. Tables are truncated
after every test, so tests stay isolated without paying to re-run migrations per test.
"""
from __future__ import annotations

import subprocess

import pytest
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.engine import Connection

from alembic import command
from magenta.config import repo_root
from magenta.db import get_engine

TENANT_A = "org_test_aaa"
TENANT_B = "org_test_bbb"

_PRESERVE = {"alembic_version", "checkpoint_migrations"}
# Both are schema-version bookkeeping owned by a migration tool -- Alembic and
# LangGraph's PostgresSaver.setup() respectively -- not test data. Truncating
# either makes its owner believe the schema is unversioned and re-run its
# migrations; harmless for idempotent migrations, but turns a no-op setup()
# into repeated work and makes the version table lie about what has run.


@pytest.fixture(scope="session")
def migrated_db() -> None:
    """Run migrations once per session against DATABASE_URL."""
    cfg = Config(str(repo_root() / "backend" / "alembic.ini"))
    cfg.set_main_option("script_location", str(repo_root() / "backend" / "alembic"))
    # No downgrade-to-base here: upgrade(head) is idempotent, and DATABASE_URL may
    # point at a developer's working database. Per-test isolation comes from the
    # TRUNCATE-on-teardown in `db_conn`, not from dropping the schema.
    command.upgrade(cfg, "head")
    # Procrastinate's schema is owned by the library, not Alembic -- but unlike
    # Alembic's upgrade(head), `schema --apply` is NOT idempotent: its CREATE TYPE
    # statements have no IF NOT EXISTS, so re-running it against a working database
    # that already has the tables (confirmed empirically: "type
    # procrastinate_job_status already exists") errors out. Guard it the way
    # Alembic guards itself, by checking what's already there.
    with get_engine().connect() as conn:
        already_applied = conn.execute(
            text("SELECT to_regclass('public.procrastinate_jobs')")
        ).scalar()
    if already_applied is None:
        subprocess.run(
            ["procrastinate", "--app=magenta.jobs.app", "schema", "--apply"],
            check=True,
        )


def _truncate_all(conn: Connection) -> None:
    """Empty every application table, derived from the catalogue so this keeps working
    as later migrations add tables."""
    names = conn.execute(text(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
    )).scalars().all()
    targets = [
        n for n in names
        if n not in _PRESERVE and not n.startswith("procrastinate")
    ]
    if targets:
        quoted = ", ".join(f'"{n}"' for n in targets)
        conn.execute(text(f"TRUNCATE TABLE {quoted} RESTART IDENTITY CASCADE"))
        conn.commit()


def _seed_test_orgs(conn: Connection) -> None:
    """Both test tenants must exist before any test writes tenant-scoped rows."""
    for tenant in (TENANT_A, TENANT_B):
        conn.execute(
            text(
                'INSERT INTO "ORGANIZATIONS" ("ID", "NAME") VALUES (:id, :name) '
                'ON CONFLICT ("ID") DO NOTHING'
            ),
            {"id": tenant, "name": f"Test org {tenant}"},
        )
    conn.commit()


@pytest.fixture
def db_conn(migrated_db) -> Connection:
    """A connection whose tables are emptied after every test.

    NOT transaction-rollback isolation. SQLAlchemy Core keeps exactly ONE active
    transaction per Connection, so when code under test calls `conn.commit()` — which
    graph/tables.py, brain/bandit.py, memory/store.py and cost/cache.py all do — it
    commits the fixture's own outer transaction and a later `trans.rollback()` silently
    no-ops. Truncating on teardown is immune to that.
    """
    conn = get_engine().connect()
    _truncate_all(conn)           # first test of a run must not inherit rows
                                   # left by a prior run that crashed mid-test
    _seed_test_orgs(conn)         # AFTER the truncate, or it wipes what we just seeded
    try:
        yield conn
    finally:
        conn.rollback()          # discard any transaction the test left open
        _truncate_all(conn)
        conn.close()
