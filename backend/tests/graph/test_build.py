from datetime import datetime, timezone

import pytest
from sqlalchemy import text

from magenta.brain.uplift import Segment
from magenta.graph.build import GraphDeps, build_graph, persist_audit
from magenta.offers import Arm


class Params:
    freq_cap_days = 14
    freq_cap_max = 1
    value_cap = 40.0


def _init_state(customer, holdout=False):
    return {
        "customer_id": customer.customer_id, "campaign_id": "CAMP-A",
        "consent_flags": {"MARKETING": True},
        "risk": None, "diagnosis": None, "offer": None, "verdict": None,
        "fulfillment": None, "outcome": None, "messages": [], "audit_log": [],
        "requires_approval": False, "holdout": holdout,
    }


def _deps(customer, fakes, spy_chat, conn):
    return GraphDeps(
        risk=fakes["risk"], uplift=fakes["uplift"], bandit=fakes["bandit"],
        catalog=fakes["catalog"], oracle=fakes["oracle"], conn=conn,
        params=Params(), chat=spy_chat, load_customer=lambda cid: customer,
        checkpointer=None,
    )


def test_happy_path_persuadable_fulfilled(customer, fakes, spy_chat, monkeypatch, db_conn):
    import magenta.graph.nodes as nm
    monkeypatch.setattr(nm, "featurize", lambda c: [0.0])
    conn = db_conn
    g = build_graph(_deps(customer, fakes, spy_chat, conn))
    final = g.invoke(_init_state(customer),
                     config={"configurable": {"thread_id": "CUST-1:CAMP-A"}})
    assert final["offer"].arm is Arm.BILL_CREDIT
    assert final["fulfillment"]["STATUS"] == "FULFILLED"
    assert final["outcome"]["retained"] is True
    assert len(spy_chat.calls) == 1


def test_non_engage_exits_early_no_llm(customer, fakes, spy_chat, monkeypatch, db_conn):
    import magenta.graph.nodes as nm
    monkeypatch.setattr(nm, "classify_segment", lambda p, t: Segment.SURE_THING)
    conn = db_conn
    g = build_graph(_deps(customer, fakes, spy_chat, conn))
    final = g.invoke(_init_state(customer),
                     config={"configurable": {"thread_id": "CUST-1:CAMP-A"}})
    assert final["risk"].engage is False
    assert final["diagnosis"] is None
    assert final["offer"] is None
    assert len(spy_chat.calls) == 0   # cost firewall: no LLM on non-engage


def test_guardrail_reject_stops_before_act(customer, fakes, spy_chat, monkeypatch, db_conn):
    import magenta.graph.nodes as nm
    monkeypatch.setattr(nm, "featurize", lambda c: [0.0])
    fakes["catalog"]._min_margin = 5.0
    fakes["catalog"]._cost = 20.0      # margin 22-20=2 < 5 => REJECT MIN_MARGIN
    conn = db_conn
    g = build_graph(_deps(customer, fakes, spy_chat, conn))
    final = g.invoke(_init_state(customer),
                     config={"configurable": {"thread_id": "CUST-1:CAMP-A"}})
    assert final["verdict"].decision == "REJECT"
    assert final["fulfillment"] is None
    n = conn.execute(text('SELECT count(*) FROM "FULFILLMENTS"')).scalar()
    assert n == 0


def test_holdout_shadow_no_fulfill_row_audit_present(customer, fakes, spy_chat, monkeypatch, db_conn):
    import magenta.graph.nodes as nm
    monkeypatch.setattr(nm, "featurize", lambda c: [0.0])
    conn = db_conn
    g = build_graph(_deps(customer, fakes, spy_chat, conn))
    final = g.invoke(_init_state(customer, holdout=True),
                     config={"configurable": {"thread_id": "CUST-1:CAMP-A"}})
    assert final["fulfillment"]["status"] == "SHADOW"
    n = conn.execute(text('SELECT count(*) FROM "FULFILLMENTS"')).scalar()
    assert n == 0
    nodes = [a["NODE"] for a in final["audit_log"]]
    assert "ACT" in nodes and "OUTCOME" in nodes


def test_idempotent_replay_one_row(customer, fakes, spy_chat, monkeypatch, db_conn):
    import magenta.graph.nodes as nm
    monkeypatch.setattr(nm, "featurize", lambda c: [0.0])
    conn = db_conn
    g = build_graph(_deps(customer, fakes, spy_chat, conn))
    cfg = {"configurable": {"thread_id": "CUST-1:CAMP-A"}}
    g.invoke(_init_state(customer), config=cfg)
    g.invoke(_init_state(customer), config=cfg)   # same thread + key
    n = conn.execute(text('SELECT count(*) FROM "FULFILLMENTS"')).scalar()
    assert n == 1


def test_no_hidden_leak_across_full_invoke(customer, fakes, spy_chat, monkeypatch, db_conn):
    import magenta.graph.nodes as nm
    monkeypatch.setattr(nm, "featurize", lambda c: [0.0])
    conn = db_conn
    g = build_graph(_deps(customer, fakes, spy_chat, conn))
    g.invoke(_init_state(customer),
             config={"configurable": {"thread_id": "CUST-1:CAMP-A"}})
    joined = " ".join(spy_chat.prompts).lower()
    for tok in ["theta_churn", "theta_price", "persuadable_segment", "competitor_pull"]:
        assert tok not in joined


def test_persist_audit_three_arg_call_regression(db_conn):
    """Regression: persist_audit(conn, audit_log) was the pre-tenant_id two-arg
    form. cli.py, api/routes_stream.py, and graph/policy.py all still called it
    that way after the signature grew a tenant_id parameter -- silently binding
    audit_log's value to tenant_id and TypeError'ing on the missing third arg.
    Pin the real 3-arg contract directly against Postgres."""
    audit_log = [{"NODE": "SENSE", "CUSTOMER_ID": "CUST-STALE-CALLER",
                  "TS": datetime.now(timezone.utc).isoformat(),
                  "PAYLOAD": '{"engage": false}'}]
    persist_audit(db_conn, "org_regression", audit_log)  # would TypeError on the old 2-arg form
    row = db_conn.execute(
        text('SELECT "TENANT_ID" FROM "AUDIT_LOG" WHERE "CUSTOMER_ID" = :c'),
        {"c": "CUST-STALE-CALLER"},
    ).mappings().first()
    assert row is not None and row["TENANT_ID"] == "org_regression"
