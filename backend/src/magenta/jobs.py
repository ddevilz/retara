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
# procrastinate CLI." So the app stays async. Sync call sites still defer synchronously
# through this same app: PsycopgConnector supports `Task.configure(connection=...)`,
# which runs the job INSERT on a caller-supplied psycopg connection instead of the
# connector's own pool — that's the atomic-enqueue mechanism, not a second app. (A
# `sync_app = app.with_connector(SyncPsycopgConnector(...))` looks tempting but is a
# trap: `with_connector` is deprecated since Procrastinate 2.14 because the tasks it
# returns still point back at the *original* app's blueprint, so `sync_app.tasks[...]
# .defer(...)` silently routes through the unopened async app and raises AppNotOpen.)
# Call `app.open()` once at process startup before any `.defer()` call — `App.open()`/
# `.close()` are themselves sync methods even though the connector is async.
