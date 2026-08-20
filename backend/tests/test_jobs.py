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
