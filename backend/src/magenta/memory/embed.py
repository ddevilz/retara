"""Local CPU sentence-embedding wrapper (Lab 12 Task 12.1).

Free, offline, deterministic embeddings for temporal customer memory (§5.9).
Owner rule: all imports at module top -- sentence_transformers is imported
here unconditionally (never lazily inside a function), even though the first
call constructs a ~80MB model download. That is a one-time package/model
fetch, not per-test network I/O -- the model is cached under
~/.cache/torch/sentence_transformers after the first run.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class LocalEmbedder:
    """CPU sentence-transformers embeddings. Free, offline, deterministic."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vecs, dtype=np.float32)

    def similarity(self, a: str, b: str) -> float:
        v = self.encode([a, b])
        return float(np.dot(v[0], v[1]))  # normalized -> dot == cosine
