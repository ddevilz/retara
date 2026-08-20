"""Procrastinate job definitions.

Postgres-backed rather than Celery/Redis for one reason that matters here: `defer()`
accepts an external connection, so a job commits in the same transaction as the write
that triggered it. With a separate broker, `create org -> enqueue provisioning` is a
dual write, and both orderings are broken — enqueue-then-commit races a worker against
an uncommitted row, commit-then-enqueue loses the job if the process dies between.

Procrastinate owns its own schema via `procrastinate schema --apply`. It is deliberately
NOT wrapped in an Alembic revision: its migrations ship with the library and are versioned
against it, so vendoring them into our history would break on every upgrade.
"""
from __future__ import annotations

import procrastinate

from magenta.db import database_url


def procrastinate_conninfo() -> str:
    """SQLAlchemy needs `postgresql+psycopg://`; libpq rejects the dialect suffix."""
    return database_url().replace("postgresql+psycopg://", "postgresql://")


app = procrastinate.App(
    connector=procrastinate.PsycopgConnector(conninfo=procrastinate_conninfo())
)
# `procrastinate` (CLI: schema/worker) refuses a sync connector — confirmed by running
# `procrastinate --app=magenta.jobs.app schema --apply` against this app, which errored
# "The connector provided by the app is not async. Please use an async connector for the
# procrastinate CLI." So the app itself stays async, and every sync call site (SQLAlchemy
# Core, sync route handlers) defers jobs through this sync-connector view instead.
sync_app = app.with_connector(
    procrastinate.SyncPsycopgConnector(conninfo=procrastinate_conninfo())
)
