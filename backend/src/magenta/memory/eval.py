"""Mini temporal-retrieval eval (Lab 12 Task 12.6) -- a number, not a vibe.

Builds N synthetic customer timelines each containing one superseded fact
and one current fact for the same (subject, relation), via
CustomerMemory.consolidate, then answers a temporal query ("what plan are
they on NOW?") two ways and scores both:

- temporal_accuracy: semantic_recall(query) filtered to the currently-open
  (VALID_TO IS NULL) edge must name the CURRENT object, not the stale one.
  This is the point of the exercise -- raw embedding similarity alone can't
  tell "mobile_s" and "mobile_l" apart in time, only consolidation's
  open/closed bookkeeping can, so recall + consolidation together must win.
- consolidation_conflict_resolution_rate: the prior edge must have been
  closed (VALID_TO == the new fact's valid_from) by consolidate() itself.

No LLM anywhere (spec §5.9 / arXiv 2606.01435): recall is a plain cosine
lookup and consolidation is a plain SQL update, both already tested in
isolation (Tasks 12.3/12.4) -- this eval just exercises them together on a
seeded synthetic batch and reports the two rates as a deterministic number.
"""
from __future__ import annotations

import uuid

import numpy as np
from sqlalchemy import text

from magenta.db import get_conn
from magenta.memory.embed import LocalEmbedder
from magenta.memory.store import CustomerMemory

_PLAN_PAIRS: list[tuple[str, str]] = [
    ("mobile_s", "mobile_l"),
    ("mobile_m", "mobile_xl"),
    ("basic", "premium"),
    ("standard", "unlimited"),
]

_QUERY = "what plan is the customer on right now?"


def run_memory_eval(n: int = 50, seed: int = 7) -> str:
    rng = np.random.default_rng(seed)

    # Table is real Postgres now (persists across runs), unlike the old
    # per-call :memory: SQLite db -- run under a throwaway tenant so
    # eval writes never accumulate in real tenant data, and clean it up
    # in `finally` below.
    eval_tenant = f"eval-{uuid.uuid4().hex[:8]}"

    # `with get_conn() as conn:` guarantees the pooled connection is
    # returned even on exception. CustomerMemory.add_edge commits
    # internally, so a rollback-based cleanup on an unclosed connection
    # would not undo anything -- an explicit close (and explicit DELETE
    # below) is the only way to avoid leaving the connection idle in
    # transaction (which deadlocks a later TRUNCATE) and leaving eval
    # rows behind in MEMORY_EDGES.
    with get_conn() as conn:
        mem = CustomerMemory(conn, eval_tenant, embedder=LocalEmbedder())
        try:
            temporal_correct = 0
            conflict_resolved = 0
            for i in range(n):
                cid = f"EVAL-{i}"
                old_plan, new_plan = _PLAN_PAIRS[int(rng.integers(len(_PLAN_PAIRS)))]
                old_date = f"2026-{int(rng.integers(1, 6)):02d}-01"
                new_date = f"2026-{int(rng.integers(6, 12)):02d}-01"

                mem.consolidate(cid, "customer", "PLAN_IS", old_plan, old_date)
                mem.consolidate(cid, "customer", "PLAN_IS", new_plan, new_date)

                # recall + consolidation together: pull related edges semantically,
                # then trust ONLY the currently-open one for a temporal answer.
                recalled = mem.semantic_recall(cid, _QUERY, k=5)
                current_candidates = [e for e in recalled if e.valid_to is None]
                if current_candidates and current_candidates[0].object == new_plan:
                    temporal_correct += 1

                # consolidation in isolation: did the stale edge get closed correctly?
                timeline = mem.timeline(cid)
                closed = [e for e in timeline if e.relation == "PLAN_IS" and e.valid_to is not None]
                if (len(closed) == 1 and closed[0].object == old_plan
                        and closed[0].valid_to.startswith(new_date)):
                    conflict_resolved += 1
        finally:
            conn.execute(text('DELETE FROM "MEMORY_EDGES" WHERE "TENANT_ID" = :t'), {"t": eval_tenant})
            conn.commit()

    temporal_accuracy = temporal_correct / n
    consolidation_conflict_resolution_rate = conflict_resolved / n
    return (
        f"temporal_accuracy={temporal_accuracy:.4f}  "
        f"consolidation_conflict_resolution_rate={consolidation_conflict_resolution_rate:.4f}  "
        f"(n={n}, seed={seed})"
    )
