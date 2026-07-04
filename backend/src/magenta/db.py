"""Shared SQLite connection helper for the app DB (data/magenta.db, gitignored).

Every module from Lab 6 onward that needs the app-wide DB (as opposed to a
throwaway `:memory:` connection in tests) opens it through here so the path
resolution (repo-root-anchored via `magenta.config.data_dir`, NOT
cwd-relative) lives in exactly one place.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from magenta.config import data_dir


def get_conn(path: str | Path | None = None) -> sqlite3.Connection:
    """Open the shared app DB. Default: <repo_root>/data/magenta.db.

    Sets row_factory = sqlite3.Row so callers can do row["COLUMN"] access
    (matches the dict-like access `magenta.graph.tables` helpers rely on).
    """
    db_path = Path(path) if path is not None else data_dir() / "magenta.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn
