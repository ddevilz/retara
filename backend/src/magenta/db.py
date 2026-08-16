"""Postgres connection layer.

SQLAlchemy Core, not the ORM: this repo writes deliberate SQL and keeps doing so.
`get_conn()` keeps its name from the SQLite era so call sites read the same, but it
now returns a SQLAlchemy `Connection` — callers use `text()` with named parameters
(`:name`), not `?` placeholders, and `conn.commit()` still applies.
"""
from __future__ import annotations

import os
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import Connection


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Local dev: "
            "postgresql+psycopg://magenta:magenta@localhost:5433/magenta"
        )
    return url


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """One pooled engine per process. `pool_pre_ping` survives Postgres restarts
    and Railway's connection recycling."""
    return create_engine(database_url(), pool_pre_ping=True, future=True)


def get_conn() -> Connection:
    """A new connection from the pool. Caller closes it (use as a context manager)."""
    return get_engine().connect()
