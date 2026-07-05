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
"""
from __future__ import annotations

import json
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Optional

from magenta.api.schemas import (
    AuditRow,
    CustomerSummary,
    Rung,
    ScorecardData,
    Scorecards,
)
from magenta.config import data_dir
from magenta.sim.population import Customer, generate_population

DATA_DIR: Path = data_dir()
SCORECARDS_PATH = DATA_DIR / "scorecards.json"
DB_PATH = DATA_DIR / "magenta.db"

# Population seed/size used for the demo customer directory. Must match the
# seed the graph/experiment use so IDs line up with AUDIT_LOG rows.
DEMO_POP_N = 2000
DEMO_POP_SEED = 7


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


def _customer_to_summary(c: Customer) -> CustomerSummary:
    """Project a Customer pydantic (observable-only) into the API summary.

    Field names here are pinned to the real magenta.sim.population.Customer
    model (verified against source, not guessed): monthly_charge (singular),
    nps_last, support_tickets_90d, clv_estimate, gross_margin_monthly.
    data_util_ratio / dropped_call_rate don't exist as raw fields — derived
    the same way magenta.brain.features._data_util_ratio does for the
    former; dropped_call_rate is dropped_calls_30d normalized to a per-day
    rate (the raw field is a 30-day count, and the frontend renders this to
    3 decimal places, so a fractional per-day rate is the sensible reading).
    """
    allowance = c.data_allowance_gb
    data_util_ratio = 0.0 if not allowance or allowance <= 0 else c.data_gb_used_p50 / allowance
    dropped_call_rate = c.dropped_calls_30d / 30.0

    return CustomerSummary(
        customer_id=c.customer_id,
        tenure_months=c.tenure_months,
        contract=c.contract,
        monthly_charges=c.monthly_charge,
        total_charges=c.total_charges,
        data_util_ratio=data_util_ratio,
        dropped_call_rate=dropped_call_rate,
        nps=float(c.nps_last) if c.nps_last is not None else None,
        support_tickets=c.support_tickets_90d,
        contract_end_days=c.contract_end_days,
        clv=c.clv_estimate,
        gross_margin=c.gross_margin_monthly,
    )


@lru_cache(maxsize=1)
def _demo_population() -> list[CustomerSummary]:
    # generate_population returns (customers, hidden_store); the hidden half
    # is simulator-private and is discarded here without ever being read.
    customers, _hidden = generate_population(DEMO_POP_N, DEMO_POP_SEED)
    return [_customer_to_summary(c) for c in customers]


def list_customers(limit: int = 50, search: str = "") -> list[CustomerSummary]:
    rows = _demo_population()
    if search:
        s = search.lower()
        rows = [r for r in rows if s in r.customer_id.lower()]
    return rows[: max(0, limit)]


def get_customer(customer_id: str) -> Optional[CustomerSummary]:
    for r in _demo_population():
        if r.customer_id == customer_id:
            return r
    return None


def _open_db() -> Optional[sqlite3.Connection]:
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def audit_rows(customer_id: str, limit: int = 50) -> list[AuditRow]:
    conn = _open_db()
    if conn is None:
        return []
    try:
        cur = conn.execute(
            """
            SELECT ID, TS, CUSTOMER_ID, NODE, PAYLOAD
            FROM AUDIT_LOG
            WHERE CUSTOMER_ID = ?
            ORDER BY ID DESC
            LIMIT ?
            """,
            (customer_id, limit),
        )
        out: list[AuditRow] = []
        for row in cur.fetchall():
            try:
                decision = json.loads(row["PAYLOAD"]) if row["PAYLOAD"] else {}
            except (json.JSONDecodeError, TypeError):
                decision = {"raw": row["PAYLOAD"]}
            rationale = str(decision.get("rationale") or decision.get("narrative") or "")
            holdout = bool(decision.get("holdout", False))
            out.append(
                AuditRow(
                    id=row["ID"],
                    ts=str(row["TS"]),
                    customer_id=row["CUSTOMER_ID"],
                    node=row["NODE"],
                    decision=decision,
                    rationale=rationale,
                    holdout=holdout,
                )
            )
        return out
    finally:
        conn.close()
