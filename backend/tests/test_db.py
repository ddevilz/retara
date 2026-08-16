"""DATABASE_URL-driven engine. No :memory: SQLite anywhere in this repo's tests."""
import pytest
from sqlalchemy import text

from magenta.db import database_url, get_conn, get_engine


def test_database_url_requires_env(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    get_engine.cache_clear()
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        database_url()


def test_get_conn_executes_against_postgres():
    with get_conn() as conn:
        assert conn.execute(text("SELECT 1")).scalar() == 1


def test_engine_is_postgres_not_sqlite():
    assert get_engine().dialect.name == "postgresql"
