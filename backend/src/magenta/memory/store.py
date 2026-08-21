"""MEMORY_EDGES temporal knowledge graph (Lab 12 Task 12.2+).

Each row is a (subject, relation, object) fact scoped to one customer, valid
over [VALID_FROM, VALID_TO). VALID_TO IS NULL means "still current". This is
the agent's episodic/semantic memory across sessions -- entirely OBSERVABLE
content (offers given, outcomes, stated preferences), never L1 hidden
simulator state (spec anti-circularity rule).

Table + column names ALL_CAPS per repo convention. Schema lives in Alembic
(0001_baseline_schema) -- no CREATE TABLE here.
"""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.engine import Connection


class MemoryEdge(BaseModel):
    subject: str
    relation: str
    object: str
    valid_from: str
    valid_to: str | None = None


def _to_dt(value: str | datetime) -> datetime:
    return value if isinstance(value, datetime) else datetime.fromisoformat(value)


def _to_iso_str(value: datetime | date | None) -> str | None:
    """VALID_FROM/VALID_TO are TIMESTAMPTZ. Inputs to add_edge/consolidate may
    be date-only ("2026-05-01") or full-timestamp ISO strings (the live graph
    path uses `_now_iso()`, which includes time-of-day); Postgres stores full
    precision either way. Read back the full ISO string -- never truncate to
    the date part, or same-day events become indistinguishable."""
    if value is None:
        return None
    return value.isoformat()


def _to_iso_str_required(value: datetime | date | None) -> str:
    """Same as `_to_iso_str`, for VALID_FROM specifically: the column is
    TIMESTAMPTZ NOT NULL (see alembic/versions/0001_baseline_schema.py), so a
    None here means the row violates that constraint -- fail loudly."""
    if value is None:
        raise ValueError("MEMORY_EDGES.VALID_FROM was NULL despite the NOT NULL constraint")
    return value.isoformat()


class CustomerMemory:
    """Temporal KG over one shared Postgres connection, scoped to one tenant.
    `embedder` is optional -- without one, `add_edge`/`timeline` still work."""

    def __init__(self, conn: Connection, tenant_id: str, embedder=None):
        self.conn = conn
        self.tenant_id = tenant_id
        self.embedder = embedder

    def add_edge(self, customer_id: str, subject: str, relation: str, obj: str,
                 valid_from: str, valid_to: str | None = None) -> int:
        emb = None
        if self.embedder is not None:
            emb = self.embedder.encode([f"{subject} {relation} {obj}"])[0].astype(
                np.float32
            ).tobytes()
        new_id = self.conn.execute(
            text(
                'INSERT INTO "MEMORY_EDGES" '
                '("TENANT_ID", "CUSTOMER_ID", "SUBJECT", "RELATION", "OBJECT", '
                ' "VALID_FROM", "VALID_TO", "EMBEDDING") '
                "VALUES (:tenant_id, :customer_id, :subject, :relation, :obj, "
                "        :valid_from, :valid_to, :emb) "
                'RETURNING "ID"'
            ),
            {
                "tenant_id": self.tenant_id, "customer_id": customer_id,
                "subject": subject, "relation": relation, "obj": obj,
                "valid_from": _to_dt(valid_from),
                "valid_to": _to_dt(valid_to) if valid_to is not None else None,
                "emb": emb,
            },
        ).scalar_one()
        self.conn.commit()
        return new_id

    def timeline(self, customer_id: str) -> list[MemoryEdge]:
        rows = self.conn.execute(
            text(
                'SELECT "SUBJECT", "RELATION", "OBJECT", "VALID_FROM", "VALID_TO" '
                'FROM "MEMORY_EDGES" '
                'WHERE "TENANT_ID" = :tenant_id AND "CUSTOMER_ID" = :customer_id '
                'ORDER BY "VALID_FROM" ASC, "ID" ASC'
            ),
            {"tenant_id": self.tenant_id, "customer_id": customer_id},
        ).mappings().all()
        return [
            MemoryEdge(subject=r["SUBJECT"], relation=r["RELATION"], object=r["OBJECT"],
                      valid_from=_to_iso_str_required(r["VALID_FROM"]), valid_to=_to_iso_str(r["VALID_TO"]))
            for r in rows
        ]

    def semantic_recall(self, customer_id: str, query: str, k: int = 3) -> list[MemoryEdge]:
        assert self.embedder is not None, "semantic_recall needs an embedder"
        q = self.embedder.encode([query])[0]
        rows = self.conn.execute(
            text(
                'SELECT "SUBJECT", "RELATION", "OBJECT", "VALID_FROM", "VALID_TO", "EMBEDDING" '
                'FROM "MEMORY_EDGES" '
                'WHERE "TENANT_ID" = :tenant_id AND "CUSTOMER_ID" = :customer_id '
                'AND "EMBEDDING" IS NOT NULL'
            ),
            {"tenant_id": self.tenant_id, "customer_id": customer_id},
        ).mappings().all()
        # ponytail: O(n) Python cosine scan per query, no vector index -- move
        # to pgvector (vector column + hnsw index) when a tenant's edge count
        # makes this hurt.
        scored = []
        for r in rows:
            v = np.frombuffer(r["EMBEDDING"], dtype=np.float32)
            scored.append((float(np.dot(q, v)), r))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [
            MemoryEdge(subject=r["SUBJECT"], relation=r["RELATION"], object=r["OBJECT"],
                      valid_from=_to_iso_str_required(r["VALID_FROM"]), valid_to=_to_iso_str(r["VALID_TO"]))
            for _, r in scored[:k]
        ]

    def consolidate(self, customer_id: str, subject: str, relation: str, obj: str,
                    valid_from: str) -> None:
        # deterministic recency: close any open edge with the same (subject, relation).
        # No LLM (spec §5.9 / arXiv 2606.01435) -- conflict resolution is a plain
        # SQL update, not a generation call.
        self.conn.execute(
            text(
                'UPDATE "MEMORY_EDGES" SET "VALID_TO" = :valid_from '
                'WHERE "TENANT_ID" = :tenant_id AND "CUSTOMER_ID" = :customer_id '
                'AND "SUBJECT" = :subject AND "RELATION" = :relation AND "VALID_TO" IS NULL'
            ),
            {
                "tenant_id": self.tenant_id, "valid_from": _to_dt(valid_from),
                "customer_id": customer_id, "subject": subject, "relation": relation,
            },
        )
        self.add_edge(customer_id, subject, relation, obj, valid_from, valid_to=None)
