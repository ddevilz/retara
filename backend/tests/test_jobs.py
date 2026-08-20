from sqlalchemy import text

from magenta.jobs import app, procrastinate_conninfo


def test_conninfo_strips_the_sqlalchemy_dialect(monkeypatch):
    """libpq rejects the +psycopg suffix that SQLAlchemy requires."""
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+psycopg://u:p@localhost:5433/magenta"
    )
    from magenta.db import get_engine

    get_engine.cache_clear()
    assert procrastinate_conninfo() == "postgresql://u:p@localhost:5433/magenta"


def test_conninfo_leaves_a_plain_url_alone(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5433/magenta")
    from magenta.db import get_engine

    get_engine.cache_clear()
    assert procrastinate_conninfo() == "postgresql://u:p@localhost:5433/magenta"


def test_app_is_a_procrastinate_app():
    import procrastinate

    assert isinstance(app, procrastinate.App)


@app.task(name="test_jobs.probe")
def _probe_task(x: int) -> None:
    pass


def test_defer_with_external_connection_commits_atomically(db_conn):
    """Regression for a mechanism that broke silently once already: a
    `sync_app = app.with_connector(SyncPsycopgConnector(...))` looks like the way to
    defer from sync code, but `with_connector` is deprecated precisely because the
    tasks it returns still point back at the *original* app's blueprint -- so
    `sync_app.tasks[...].defer(...)` routes through the unopened async `app` and raises
    AppNotOpen. The real mechanism is `Task.configure(connection=...)` against the
    plain async-`app`-registered task, using a caller-supplied psycopg connection so the
    job INSERT lands in the same transaction as the write that triggered it.
    """
    app.open()
    try:
        raw_conn = db_conn.connection.driver_connection
        _probe_task.configure(connection=raw_conn).defer(x=1)
        db_conn.commit()
        row = db_conn.execute(
            text("SELECT queue_name FROM procrastinate_jobs WHERE task_name = :n"),
            {"n": "test_jobs.probe"},
        ).first()
        assert row is not None
    finally:
        app.close()


def test_train_writes_both_artifacts(monkeypatch, tmp_path):
    from magenta.jobs import train_tenant_models
    from magenta.storage import risk_model_path, uplift_model_path

    monkeypatch.setenv("MAGENTA_MODEL_DIR", str(tmp_path))
    train_tenant_models("org_job_test", n=200)
    assert risk_model_path("org_job_test").exists()
    assert uplift_model_path("org_job_test").exists()


def test_train_is_deterministic_per_tenant(monkeypatch, tmp_path):
    from magenta.jobs import train_tenant_models
    from magenta.storage import risk_model_path

    monkeypatch.setenv("MAGENTA_MODEL_DIR", str(tmp_path))
    train_tenant_models("org_det_job", n=200)
    first = risk_model_path("org_det_job").read_bytes()
    train_tenant_models("org_det_job", n=200)
    assert risk_model_path("org_det_job").read_bytes() == first


def test_job_is_registered_with_a_queueing_lock():
    """Two provisioning jobs for one tenant must not train concurrently and race on
    the same artifact path."""
    from magenta.jobs import train_tenant_models_job

    assert train_tenant_models_job.queueing_lock is not None
