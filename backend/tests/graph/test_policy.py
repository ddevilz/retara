import sqlite3

import pytest

from magenta.brain.uplift import Segment
from magenta.graph.build import GraphDeps
from magenta.graph.policy import AgentPolicy
from magenta.graph.tables import init_graph_tables
from magenta.offers import Arm, OfferDecision


def _conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    init_graph_tables(c)
    return c


def _deps(customer, fakes, spy_chat, conn):
    return GraphDeps(
        risk=fakes["risk"], uplift=fakes["uplift"], bandit=fakes["bandit"],
        catalog=fakes["catalog"], oracle=fakes["oracle"], conn=conn,
        params=type("P", (), {"freq_cap_days": 14, "freq_cap_max": 1, "value_cap": 40.0})(),
        chat=spy_chat, load_customer=lambda cid: customer, checkpointer=None,
    )


def test_agent_policy_returns_offer_for_persuadable(customer, fakes, spy_chat, monkeypatch):
    import magenta.graph.nodes as nm
    monkeypatch.setattr(nm, "featurize", lambda c: [0.0])
    pol = AgentPolicy(_deps(customer, fakes, spy_chat, _conn()))
    offer = pol.decide(customer)
    assert isinstance(offer, OfferDecision)
    assert offer.arm is Arm.BILL_CREDIT


def test_agent_policy_none_for_non_engage(customer, fakes, spy_chat, monkeypatch):
    import magenta.graph.nodes as nm
    monkeypatch.setattr(nm, "classify_segment", lambda p, t: Segment.LOST_CAUSE)
    pol = AgentPolicy(_deps(customer, fakes, spy_chat, _conn()))
    assert pol.decide(customer) is None
    assert len(spy_chat.calls) == 0


def test_agent_policy_none_on_reject(customer, fakes, spy_chat, monkeypatch):
    import magenta.graph.nodes as nm
    monkeypatch.setattr(nm, "featurize", lambda c: [0.0])
    fakes["catalog"]._min_margin = 5.0
    fakes["catalog"]._cost = 20.0     # margin 2 < 5 => REJECT
    pol = AgentPolicy(_deps(customer, fakes, spy_chat, _conn()))
    assert pol.decide(customer) is None
