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
