"""Local CPU sentence-embedding wrapper (Lab 12 Task 12.1).

Free, offline, deterministic embeddings for temporal customer memory (§5.9).
Owner rule: all imports at module top -- sentence_transformers is imported
here unconditionally (never lazily inside a function), even though the first
call constructs a ~80MB model download. That is a one-time package/model
fetch, not per-test network I/O -- the model is cached under
~/.cache/torch/sentence_transformers after the first run.
"""
from __future__ import annotations

import os

# CRASH WORKAROUND (verified on this machine, macOS/Apple Silicon): when a
# process that already imported lightgbm (magenta.brain.risk/uplift, and
# transitively the whole tests/conftest.py -> magenta.graph.build chain)
# later runs a torch forward pass (SentenceTransformer.encode ->
# torch.nn.functional.layer_norm), the two libraries' OpenMP thread pools
# collide and SIGSEGV inside the native layer_norm kernel -- reproduced with
# a minimal `import lightgbm; LocalEmbedder().encode(...)` script. Pinning
# OMP_NUM_THREADS=1 before torch initializes its own thread pool avoids the
# collision; must be set before `sentence_transformers`/torch import below.
# Negligible cost here: MiniLM-L6 CPU inference on a handful of short
# sentences at a time (never a large batch).
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import tqdm
from sentence_transformers import SentenceTransformer

# Workaround for a known tqdm/torch interpreter-shutdown segfault: tqdm's
# background monitor thread (one per progress bar) can outlive the process
# that spawned it when several SentenceTransformer instances are created in
# one run (e.g. multiple tests, or CLI `memory show` + `memory eval` in the
# same process). Setting monitor_interval=0 disables that thread entirely --
# a documented tqdm mitigation, harmless here since nothing depends on the
# monitor's stalled-iterator warnings.
tqdm.tqdm.monitor_interval = 0


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
