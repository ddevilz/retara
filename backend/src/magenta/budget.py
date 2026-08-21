"""Per-tenant LLM token metering and budget enforcement."""
from __future__ import annotations

from sqlalchemy import text

from magenta.db import get_conn
from magenta.logging_config import get_logger
from magenta.tenancy import BoundedTTLCache

logger = get_logger(__name__)

# Budget state is checked before every LLM call, and LLM calls happen in cohort loops.
# Aggregating LLM_USAGE per call would put a query in the hottest path in the system, so
# the answer is cached briefly. A tenant can overshoot by at most one TTL window of
# calls, which is an acceptable trade for not querying per call.
_BUDGET_CACHE = BoundedTTLCache(maxsize=64, ttl_seconds=60)


class BudgetExceeded(Exception):
    """This tenant is over its monthly token budget."""


def record_usage(tenant_id: str, role: str, model: str,
                 tokens_in: int, tokens_out: int) -> None:
    with get_conn() as conn:
        conn.execute(
            text(
                'INSERT INTO "LLM_USAGE" '
                '("TENANT_ID", "ROLE", "MODEL", "TOKENS_IN", "TOKENS_OUT") '
                "VALUES (:tenant_id, :role, :model, :tin, :tout)"
            ),
            {"tenant_id": tenant_id, "role": role, "model": model,
             "tin": tokens_in, "tout": tokens_out},
        )
        conn.commit()


def tokens_used_this_month(tenant_id: str) -> int:
    with get_conn() as conn:
        return conn.execute(
            text(
                'SELECT COALESCE(SUM("TOKENS_IN" + "TOKENS_OUT"), 0) '
                'FROM "LLM_USAGE" '
                'WHERE "TENANT_ID" = :tenant_id '
                "AND \"TS\" >= date_trunc('month', NOW())"
            ),
            {"tenant_id": tenant_id},
        ).scalar_one()


def is_over_budget(tenant_id: str) -> bool:
    cached = _BUDGET_CACHE.get(tenant_id)
    if cached is not None:
        return cached

    with get_conn() as conn:
        budget = conn.execute(
            text('SELECT "MONTHLY_TOKEN_BUDGET" FROM "ORGANIZATIONS" WHERE "ID" = :id'),
            {"id": tenant_id},
        ).scalar()

    over = False if budget is None else tokens_used_this_month(tenant_id) >= budget
    if over:
        logger.warning("budget.exceeded", tenant_id=tenant_id, budget=budget)
    _BUDGET_CACHE.put(tenant_id, over)
    return over
