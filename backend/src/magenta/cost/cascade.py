"""Confidence cascade: cheap-first, escalate to the large role on low
confidence. `chat_fn`/`confidence_fn` are injected so callers (and tests)
never need a real network round-trip -- see `magenta.graph.batch_diagnose`
for the production wiring (role names "cheap"/"large" map to models.yaml).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class CascadeResult:
    answer: str
    role_used: str
    escalated: bool


def cascade(
    messages,
    chat_fn: Callable[[str, list], str],
    confidence_fn: Callable[[str], float],
    cheap: str = "cheap",
    large: str = "large",
    tau: float = 0.6,
) -> CascadeResult:
    ans = chat_fn(cheap, messages)
    if confidence_fn(ans) >= tau:
        return CascadeResult(ans, cheap, False)
    return CascadeResult(chat_fn(large, messages), large, True)
