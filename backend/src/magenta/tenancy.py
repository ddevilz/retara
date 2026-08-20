"""Per-tenant seeds and the bounded cache that holds per-tenant runtime state.

`tenant_seed` uses sha256, never `hash()`: Python salts `hash()` per process, so a
seed derived from it would change on every restart and break this repo's
"same seed => identical output" guarantee.
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any, Callable


def tenant_seed(tenant_id: str) -> int:
    """Deterministic 32-bit seed for a tenant's simulated sandbox."""
    digest = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


class BoundedTTLCache:
    """LRU with a TTL, guarding two different failure modes.

    maxsize bounds memory — each entry can hold two LightGBM model sets and a
    population. ttl_seconds bounds staleness — the Phase 1.4 worker retrains in a
    separate process and cannot reach into this one to invalidate.

    Note: on_evict callbacks run while self._lock is held. A callback must never
    call back into this cache (get/put/invalidate/clear) or it will deadlock —
    threading.Lock is not reentrant.
    """

    def __init__(
        self,
        maxsize: int,
        ttl_seconds: float,
        on_evict: Callable[[Any], None] | None = None,
    ) -> None:
        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self._on_evict = on_evict
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            stored_at, value = entry
            if time.monotonic() - stored_at > self._ttl:
                del self._data[key]
                if self._on_evict is not None:
                    self._on_evict(value)
                return None
            self._data.move_to_end(key)
            return value

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            # If key already exists, evict the old value before replacing it
            if key in self._data:
                _, old_value = self._data[key]
                if self._on_evict is not None:
                    self._on_evict(old_value)

            self._data[key] = (time.monotonic(), value)
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                _, evicted_value = self._data.popitem(last=False)
                if self._on_evict is not None:
                    self._on_evict(evicted_value)

    def invalidate(self, key: str) -> None:
        with self._lock:
            entry = self._data.pop(key, None)
            if entry is not None and self._on_evict is not None:
                _, value = entry
                self._on_evict(value)

    def clear(self) -> None:
        with self._lock:
            if self._on_evict is not None:
                for _, value in self._data.values():
                    self._on_evict(value)
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
