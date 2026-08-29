from unittest.mock import MagicMock

import pytest
from sqlalchemy import text

from magenta.brain.risk import Band, Driver
from magenta.brain.uplift import Segment
from magenta.budget import BudgetExceeded, is_over_budget, record_usage, tokens_used_this_month
from magenta.context import set_tenant
from magenta.graph.nodes import diagnose
from magenta.graph.state import RiskUpliftReport, Timing
from magenta.llm import chat
from tests.db_fixtures import TENANT_A, TENANT_B


def test_usage_accumulates_per_tenant(db_conn):
    record_usage(TENANT_A, "cheap", "llama-3.1-8b", 100, 50)
    record_usage(TENANT_A, "large", "llama-3.3-70b", 200, 100)
    assert tokens_used_this_month(TENANT_A) == 450


def test_usage_is_tenant_isolated(db_conn):
    record_usage(TENANT_A, "cheap", "llama-3.1-8b", 100, 50)
    assert tokens_used_this_month(TENANT_B) == 0


def test_unknown_tenant_has_zero_usage(db_conn):
    assert tokens_used_this_month("org_never_seen") == 0


def test_over_budget_raises_before_calling_the_provider(db_conn, monkeypatch):
    """Refuse before spending, not after."""
    from magenta.budget import _BUDGET_CACHE

    db_conn.execute(
        text('UPDATE "ORGANIZATIONS" SET "MONTHLY_TOKEN_BUDGET" = 10 WHERE "ID" = :id'),
        {"id": TENANT_A},
    )
    db_conn.commit()
    record_usage(TENANT_A, "cheap", "m", 100, 100)

    set_tenant(TENANT_A)
    client = MagicMock()
    monkeypatch.setattr("magenta.llm.get_client", lambda: client)
    _BUDGET_CACHE.clear()

    with pytest.raises(BudgetExceeded):
        chat("cheap", [{"role": "user", "content": "hi"}])
    client.chat.completions.create.assert_not_called()


def test_null_budget_means_unlimited(db_conn):
    from magenta.budget import _BUDGET_CACHE

    _BUDGET_CACHE.clear()
    record_usage(TENANT_A, "cheap", "m", 10_000_000, 10_000_000)
    assert is_over_budget(TENANT_A) is False


def test_graph_degrades_to_no_action_when_over_budget(mem_deps_factory, monkeypatch):
    """The claim under test: budget exhaustion reuses the EXISTING degradation paths.
    If this fails, widen the handler in graph/nodes.py::diagnose -- do not add a
    parallel budget-specific code path."""
    deps = mem_deps_factory()
    monkeypatch.setattr(
        deps.chat, "chat_structured", MagicMock(side_effect=BudgetExceeded("over"))
    )
    # sense() would have populated "risk" before routing here; diagnose() requires
    # it be present regardless of budget state, so inject a minimal report.
    state = {
        "customer_id": "CUST_0001",
        "risk": RiskUpliftReport(
            p_churn=0.72, band=Band.HIGH,
            drivers=[Driver(feature="OVERAGE_EVENTS", label="Overage events",
                            shap_value=0.31, direction="UP")],
            tau_hat=0.18, segment=Segment.PERSUADABLE, engage=True, timing=Timing.ACT_NOW,
        ),
    }
    result = diagnose(state, deps=deps)
    assert result.get("diagnosis") is None or result.get("offer") is None
