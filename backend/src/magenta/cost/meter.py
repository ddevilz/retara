"""CostMeter: the numbers behind the cost/rate levers (semantic cache +
confidence cascade) -- routed-cheap / cache-hit / escalation rates over a
run, printed by `magenta cost report`.
"""
from __future__ import annotations


class CostMeter:
    def __init__(self):
        self.n = 0
        self.cheap = 0
        self.hits = 0
        self.esc = 0

    def record(self, role_used: str, cache_hit: bool, escalated: bool) -> None:
        self.n += 1
        if cache_hit:
            self.hits += 1
        if role_used == "cheap":
            self.cheap += 1
        if escalated:
            self.esc += 1

    def report(self) -> dict:
        n = max(self.n, 1)
        return {
            "total_decisions": self.n,
            "pct_routed_cheap": self.cheap / n,
            "cache_hit_rate": self.hits / n,
            "escalation_rate": self.esc / n,
        }
