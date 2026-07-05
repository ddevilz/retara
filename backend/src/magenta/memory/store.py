"""MEMORY_EDGES temporal knowledge graph (Lab 12 Task 12.2+).

Each row is a (subject, relation, object) fact scoped to one customer, valid
over [VALID_FROM, VALID_TO). VALID_TO IS NULL means "still current". This is
the agent's episodic/semantic memory across sessions -- entirely OBSERVABLE
content (offers given, outcomes, stated preferences), never L1 hidden
simulator state (spec anti-circularity rule).

Table + column names ALL_CAPS per repo convention.
"""
from __future__ import annotations

import sqlite3

import numpy as np
from pydantic import BaseModel


class MemoryEdge(BaseModel):
    subject: str
    relation: str
    object: str
    valid_from: str
    valid_to: str | None = None


class CustomerMemory:
    """Temporal KG over one shared SQLite connection. `embedder` is optional --
    without one, `add_edge`/`timeline` still work."""

    def __init__(self, conn: sqlite3.Connection, embedder=None):
        self.conn = conn
        self.embedder = embedder

    def init_tables(self) -> None:
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS MEMORY_EDGES (
                 ID INTEGER PRIMARY KEY AUTOINCREMENT,
                 CUSTOMER_ID TEXT NOT NULL, SUBJECT TEXT NOT NULL,
                 RELATION TEXT NOT NULL, OBJECT TEXT NOT NULL,
                 VALID_FROM TEXT NOT NULL, VALID_TO TEXT, EMBEDDING BLOB)""")
        self.conn.commit()

    def add_edge(self, customer_id, subject, relation, obj, valid_from, valid_to=None) -> int:
        emb = None
        if self.embedder is not None:
            emb = self.embedder.encode([f"{subject} {relation} {obj}"])[0].tobytes()
        cur = self.conn.execute(
            "INSERT INTO MEMORY_EDGES (CUSTOMER_ID,SUBJECT,RELATION,OBJECT,VALID_FROM,VALID_TO,EMBEDDING) "
            "VALUES (?,?,?,?,?,?,?)",
            (customer_id, subject, relation, obj, valid_from, valid_to, emb))
        self.conn.commit()
        return cur.lastrowid

    def timeline(self, customer_id) -> list[MemoryEdge]:
        rows = self.conn.execute(
            "SELECT SUBJECT,RELATION,OBJECT,VALID_FROM,VALID_TO FROM MEMORY_EDGES "
            "WHERE CUSTOMER_ID=? ORDER BY VALID_FROM ASC, ID ASC", (customer_id,)).fetchall()
        return [MemoryEdge(subject=r["SUBJECT"], relation=r["RELATION"], object=r["OBJECT"],
                           valid_from=r["VALID_FROM"], valid_to=r["VALID_TO"]) for r in rows]

    def semantic_recall(self, customer_id, query, k: int = 3) -> list[MemoryEdge]:
        assert self.embedder is not None, "semantic_recall needs an embedder"
        q = self.embedder.encode([query])[0]
        rows = self.conn.execute(
            "SELECT SUBJECT,RELATION,OBJECT,VALID_FROM,VALID_TO,EMBEDDING FROM MEMORY_EDGES "
            "WHERE CUSTOMER_ID=? AND EMBEDDING IS NOT NULL", (customer_id,)).fetchall()
        scored = []
        for r in rows:
            v = np.frombuffer(r["EMBEDDING"], dtype=np.float32)
            scored.append((float(np.dot(q, v)), r))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [MemoryEdge(subject=r["SUBJECT"], relation=r["RELATION"], object=r["OBJECT"],
                           valid_from=r["VALID_FROM"], valid_to=r["VALID_TO"]) for _, r in scored[:k]]

    def consolidate(self, customer_id, subject, relation, obj, valid_from) -> None:
        # deterministic recency: close any open edge with the same (subject, relation).
        # No LLM (spec §5.9 / arXiv 2606.01435) -- conflict resolution is a plain
        # SQL update, not a generation call.
        self.conn.execute(
            "UPDATE MEMORY_EDGES SET VALID_TO=? WHERE CUSTOMER_ID=? AND SUBJECT=? AND RELATION=? AND VALID_TO IS NULL",
            (valid_from, customer_id, subject, relation))
        self.add_edge(customer_id, subject, relation, obj, valid_from, valid_to=None)
