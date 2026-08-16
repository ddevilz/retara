"""Postgres fixtures for the whole suite. Task 3 extends this file with `db_conn`
and the tenant constants; this task provides only the session-scoped migration run."""
from __future__ import annotations

import pytest
from alembic import command
from alembic.config import Config

from magenta.config import repo_root


@pytest.fixture(scope="session")
def migrated_db() -> None:
    """Run migrations once per session against DATABASE_URL."""
    cfg = Config(str(repo_root() / "backend" / "alembic.ini"))
    cfg.set_main_option("script_location", str(repo_root() / "backend" / "alembic"))
    # No downgrade-to-base here: upgrade(head) is idempotent, and DATABASE_URL may
    # point at a developer's working database. Per-test isolation comes from the
    # transaction rollback in `db_conn` (Task 3), not from dropping the schema.
    command.upgrade(cfg, "head")
