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

from magenta.brain.risk import RiskModel
from magenta.brain.training import build_training_data
from magenta.brain.uplift import UpliftModel
from magenta.db import database_url
from magenta.storage import risk_model_path, uplift_model_path
from magenta.tenancy import tenant_seed


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


def train_tenant_models(tenant_id: str, n: int = 3000) -> None:
    """Train and persist this tenant's risk and uplift models.

    A plain function, not the task itself, so the CLI and tests can call it directly
    without a queue in the way.
    """
    seed = tenant_seed(tenant_id)
    td = build_training_data(n=n, seed=seed)
    RiskModel().fit(td.customers, td.churned).save(risk_model_path(tenant_id))
    UpliftModel().fit(td.customers, td.treated, td.retained).save(
        uplift_model_path(tenant_id)
    )


@app.task(name="train_tenant_models", queueing_lock="train_tenant_models")
def train_tenant_models_job(tenant_id: str, n: int = 3000) -> None:
    train_tenant_models(tenant_id, n=n)


# ponytail: training runs in the worker process, not a subprocess. Procrastinate has no
# worker_max_tasks_per_child, so repeated LightGBM and SHAP fits can fragment worker
# memory over time. At a handful of tenants that is theoretical — restart the worker on
# deploy. If RSS actually climbs, run the worker with --one-shot under a supervisor
# before reaching for subprocess isolation.
