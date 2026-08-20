"""Read-only helpers the API uses to fetch persisted state.

Everything here is import-and-call over the existing magenta package + the
SQLite app tables. No writes. No hidden simulator state ever leaves this module
(we only ever read the observable Customer projection).

Two deviations from the Task 10.2 brief snippet, found while wiring this up
(see backend/tests/api/test_customers.py + data-flow checks):

1. ``generate_population(n, seed)`` returns ``(list[Customer], HiddenStore)``,
   not a bare list. The HiddenStore half is discarded immediately here and
   never touched again -- it must never reach a CustomerSummary.
2. ``AUDIT_LOG`` (backend/src/magenta/graph/tables.py) has columns
   ``ID, NODE, CUSTOMER_ID, TS, PAYLOAD`` -- there is no ``DECISION_JSON``,
   ``RATIONALE``, or ``HOLDOUT`` column. Every node's audit entry stores its
   whole payload dict as a single JSON blob in ``PAYLOAD``, and the payload
   shape differs per node (see graph/nodes.py `_audit`). ``decision`` below
   is that parsed payload; ``rationale``/``holdout`` are best-effort keys
   pulled out of it (present on some node payloads, e.g. OUTCOME sets
   "holdout", but not guaranteed on all) since no node persists a dedicated
   rationale string today.

Postgres note (Task 9): AUDIT_LOG now lives in Postgres (Alembic
0001_baseline_schema), TENANT_ID-scoped. PAYLOAD is JSONB -- psycopg
returns a dict, so there is no json.loads here; TS is TIMESTAMPTZ -- a
datetime, so .isoformat() rather than str(). ``_open_db``/``DB_PATH`` (a
second, hardcoded connection path to the same physical file get_conn()
already opens) are gone -- one connection path only.
"""
from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import text

from magenta.api.schemas import (
    AuditRow,
    Rung,
    ScorecardData,
    Scorecards,
)
from magenta.config import data_dir
from magenta.db import get_conn

DATA_DIR: Path = data_dir()
SCORECARDS_PATH = DATA_DIR / "scorecards.json"


def load_scorecards() -> Scorecards:
    if not SCORECARDS_PATH.exists():
        # Empty ladder rather than 500 — dashboard shows "run the ablation first".
        return Scorecards(rungs=[])
    raw = json.loads(SCORECARDS_PATH.read_text())
    rungs = [
        Rung(policy=r["policy"], scorecard=ScorecardData(**r["scorecard"]))
        for r in raw.get("rungs", [])
    ]
    return Scorecards(rungs=rungs)


def audit_rows(tenant_id: str, customer_id: str, limit: int = 50) -> list[AuditRow]:
    """PAYLOAD is JSONB -- psycopg returns a dict, so there is no json.loads here."""
    with get_conn() as conn:
        rows = conn.execute(
            text(
                'SELECT "ID", "TS", "CUSTOMER_ID", "NODE", "PAYLOAD" FROM "AUDIT_LOG" '
                'WHERE "TENANT_ID" = :tenant_id AND "CUSTOMER_ID" = :customer_id '
                'ORDER BY "ID" DESC LIMIT :limit'
            ),
            {"tenant_id": tenant_id, "customer_id": customer_id, "limit": limit},
        ).mappings().all()

    out: list[AuditRow] = []
    for r in rows:
        decision = r["PAYLOAD"] or {}
        rationale = str(decision.get("rationale") or decision.get("narrative") or "")
        holdout = bool(decision.get("holdout", False))
        out.append(
            AuditRow(
                id=r["ID"],
                ts=r["TS"].isoformat(),
                customer_id=r["CUSTOMER_ID"],
                node=r["NODE"],
                decision=decision,
                rationale=rationale,
                holdout=holdout,
            )
        )
    return out
