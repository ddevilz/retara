import pytest
from sqlalchemy import text

from magenta.brain.uplift import Segment
import magenta.graph.nodes as nodes_mod
from magenta.graph.build import GraphDeps
from magenta.graph.policy import AgentPolicy
from magenta.offers import Arm, OfferDecision
from tests.db_fixtures import TENANT_A, TENANT_B


def _deps(customer, fakes, spy_chat, conn, tenant_id=TENANT_A):
    return GraphDeps(
        risk=fakes["risk"], uplift=fakes["uplift"], bandit=fakes["bandit"],
        catalog=fakes["catalog"], oracle=fakes["oracle"], conn=conn,
        params=type("P", (), {"freq_cap_days": 14, "freq_cap_max": 1, "value_cap": 40.0})(),
        chat=spy_chat, load_customer=lambda cid: customer, checkpointer=None,
        tenant_id=tenant_id,
    )


def test_agent_policy_returns_offer_for_persuadable(customer, fakes, spy_chat, monkeypatch, db_conn):
    import magenta.graph.nodes as nm
    monkeypatch.setattr(nm, "featurize", lambda c: [0.0])
    pol = AgentPolicy(_deps(customer, fakes, spy_chat, db_conn))
    offer = pol.decide(customer)
    assert isinstance(offer, OfferDecision)
    assert offer.arm is Arm.BILL_CREDIT


def test_agent_policy_none_for_non_engage(customer, fakes, spy_chat, monkeypatch, db_conn):
    """The reachable crash the reviewer identified: a customer whose engage-gate
    exits at sense() never reaches guardrail/act, so it goes straight from
    graph.invoke() to AgentPolicy.decide()'s persist_audit(self.deps.conn,
    self.deps.tenant_id, ...) call (graph/policy.py:43) -- one of the three
    call sites that still passed the old two-arg persist_audit(conn, audit_log)
    form and TypeError'd. Drives that exact path and checks the resulting
    AUDIT_LOG row is scoped to the expected tenant."""
    monkeypatch.setattr(nodes_mod, "classify_segment", lambda p, t: Segment.LOST_CAUSE)
    pol = AgentPolicy(_deps(customer, fakes, spy_chat, db_conn, tenant_id=TENANT_A))
    assert pol.decide(customer) is None
    assert len(spy_chat.calls) == 0
    row = db_conn.execute(
        text('SELECT "TENANT_ID" FROM "AUDIT_LOG" WHERE "CUSTOMER_ID" = :c AND "NODE" = :n'),
        {"c": customer.customer_id, "n": "SENSE"},
    ).mappings().first()
    assert row is not None and row["TENANT_ID"] == TENANT_A


def test_agent_policy_none_on_reject(customer, fakes, spy_chat, monkeypatch, db_conn):
    import magenta.graph.nodes as nm
    monkeypatch.setattr(nm, "featurize", lambda c: [0.0])
    fakes["catalog"]._min_margin = 5.0
    fakes["catalog"]._cost = 20.0     # margin 2 < 5 => REJECT
    pol = AgentPolicy(_deps(customer, fakes, spy_chat, db_conn))
    assert pol.decide(customer) is None


def test_thread_id_is_tenant_scoped(customer, fakes, spy_chat, monkeypatch, db_conn):
    """Same customer_id + campaign_id under two tenants must resolve to two
    different checkpointer threads. Pre-fix, thread_id = f"{customer_id}:
    {campaign_id}" collided across tenants -- with the Task 10 PostgresSaver
    that means tenant B's run RESUMES tenant A's persisted checkpoint,
    silently leaking A's offer/diagnosis/verdict into B's response."""
    monkeypatch.setattr(nodes_mod, "featurize", lambda c: [0.0])
    seen_thread_ids = []
    for tenant in (TENANT_A, TENANT_B):
        pol = AgentPolicy(_deps(customer, fakes, spy_chat, db_conn, tenant_id=tenant))
        real_invoke = pol._graph.invoke

        def spy_invoke(state, config, _real=real_invoke):
            seen_thread_ids.append(config["configurable"]["thread_id"])
            return _real(state, config=config)

        monkeypatch.setattr(pol._graph, "invoke", spy_invoke)
        pol.decide(customer)

    assert len(seen_thread_ids) == 2
    assert seen_thread_ids[0] != seen_thread_ids[1]


def test_decide_returns_none_on_no_action_collapse(customer, fakes, spy_chat, monkeypatch, db_conn):
    """When eligible arms collapse to [NO_ACTION] (empty catalog∩diagnosis
    intersection), decide() must return None — a leaked NO_ACTION OfferDecision
    is counted by run_experiment as a real offer (39% fake-offer corruption
    measured in review)."""
    monkeypatch.setattr(nodes_mod, "featurize", lambda c: [0.0])
    fakes["bandit"]._arm = Arm.NO_ACTION
    fakes["catalog"].eligible = lambda c: [Arm.NO_ACTION]
    policy = AgentPolicy(_deps(customer, fakes, spy_chat, db_conn))
    assert policy.decide(customer) is None
