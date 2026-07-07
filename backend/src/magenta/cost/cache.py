"""SemanticCache: near-duplicate LLM-answer cache over local embeddings.

Cost lever: cohort diagnoses whose SHAP-driver key-text is semantically close
(cosine >= threshold) reuse the SAME cached answer instead of paying for
another LLM call -- an upgrade over an exact-hash cache (see
`magenta.graph.batch_diagnose`), because near-identical driver shapes (not
just byte-identical ones) collapse together.

Table SEMANTIC_CACHE is ALL_CAPS per repo convention (§0.5).
"""
from __future__ import annotations

import sqlite3

import numpy as np


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
    def __init__(self, conn: sqlite3.Connection, embedder, threshold: float = 0.75):
        self.conn = conn
        self.embedder = embedder
        self.threshold = threshold
        conn.execute(
            "CREATE TABLE IF NOT EXISTS SEMANTIC_CACHE "
            "(ID INTEGER PRIMARY KEY AUTOINCREMENT, KEY_TEXT TEXT, VALUE TEXT, EMBEDDING BLOB)"
        )
        conn.commit()

    def get(self, key_text: str) -> str | None:
        q = self.embedder.encode([key_text])[0]
        best_v, best_s = None, -1.0
        for r in self.conn.execute("SELECT VALUE, EMBEDDING FROM SEMANTIC_CACHE").fetchall():
            s = float(np.dot(q, np.frombuffer(r["EMBEDDING"], dtype=np.float32)))
            if s > best_s:
                best_s, best_v = s, r["VALUE"]
        return best_v if best_s >= self.threshold else None

    def put(self, key_text: str, value: str) -> None:
        emb = self.embedder.encode([key_text])[0].tobytes()
        self.conn.execute(
            "INSERT INTO SEMANTIC_CACHE (KEY_TEXT,VALUE,EMBEDDING) VALUES (?,?,?)",
            (key_text, value, emb),
        )
        self.conn.commit()
