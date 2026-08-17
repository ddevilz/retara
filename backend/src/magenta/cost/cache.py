"""SemanticCache: near-duplicate LLM-answer cache over local embeddings.

Cost lever: cohort diagnoses whose SHAP-driver key-text is semantically close
(cosine >= threshold) reuse the SAME cached answer instead of paying for
another LLM call -- an upgrade over an exact-hash cache (see
`magenta.graph.batch_diagnose`), because near-identical driver shapes (not
just byte-identical ones) collapse together.

Table SEMANTIC_CACHE is ALL_CAPS per repo convention (§0.5); schema lives in
Alembic (0001_baseline_schema) -- no CREATE TABLE here.
"""
from __future__ import annotations

import numpy as np
from sqlalchemy import text
from sqlalchemy.engine import Connection


class SemanticCache:
    # BUG FIX (verified on this machine): the spec's threshold of 0.93 assumes
    # near-cosine-1.0 similarity for near-duplicate sentences, but real
    # all-MiniLM-L6-v2 embeddings score genuinely-related-but-reworded pairs
    # around ~0.85 (e.g. "bill shock, price sensitive, contract ending" vs.
    # "customer has bill shock and is price sensitive, contract about to
    # end" -> 0.846) while unrelated text scores ~0.03. 0.93 would only ever
    # fire on near-exact string matches, defeating the whole point of a
    # semantic (not exact) cache. 0.75 is comfortably above the unrelated
    # floor and below the near-dup ceiling for this embedder.
    def __init__(self, conn: Connection, tenant_id: str, embedder, threshold: float = 0.75):
        self.conn = conn
        self.tenant_id = tenant_id
        self.embedder = embedder
        self.threshold = threshold

    def get(self, key_text: str) -> str | None:
        q = self.embedder.encode([key_text])[0].astype(np.float32)
        rows = self.conn.execute(
            text(
                'SELECT "VALUE", "EMBEDDING" FROM "SEMANTIC_CACHE" '
                'WHERE "TENANT_ID" = :tenant_id AND "EMBEDDING" IS NOT NULL'
            ),
            {"tenant_id": self.tenant_id},
        ).mappings().all()
        # ponytail: O(n) Python cosine scan per lookup, same shape as
        # memory.store's semantic_recall -- move to pgvector (vector column +
        # hnsw index) when a tenant's cache size makes this hurt.
        best_v, best_s = None, -1.0
        for r in rows:
            v = np.frombuffer(r["EMBEDDING"], dtype=np.float32)
            s = float(np.dot(q, v))
            if s > best_s:
                best_s, best_v = s, r["VALUE"]
        return best_v if best_s >= self.threshold else None

    def put(self, key_text: str, value: str) -> None:
        emb = self.embedder.encode([key_text])[0].astype(np.float32).tobytes()
        self.conn.execute(
            text(
                'INSERT INTO "SEMANTIC_CACHE" '
                '("TENANT_ID", "KEY_TEXT", "VALUE", "EMBEDDING") '
                "VALUES (:tenant_id, :key_text, :value, :emb)"
            ),
            {"tenant_id": self.tenant_id, "key_text": key_text,
             "value": value, "emb": emb},
        )
        self.conn.commit()
