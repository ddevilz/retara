import sqlite3

from magenta.memory.embed import LocalEmbedder
from magenta.memory.store import CustomerMemory


def _mem():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    m = CustomerMemory(conn)
    m.init_tables()
    return m


def test_add_and_timeline_ordered():
    m = _mem()
    m.add_edge("C1", "customer", "COMPLAINED_ABOUT", "coverage", "2026-03-01")
    m.add_edge("C1", "agent", "GAVE", "bill_credit", "2026-04-01")
    tl = m.timeline("C1")
    assert [e.valid_from for e in tl] == ["2026-03-01", "2026-04-01"]  # temporal order
    assert tl[1].object == "bill_credit"


def test_timeline_is_per_customer():
    m = _mem()
    m.add_edge("C1", "a", "R", "x", "2026-01-01")
    m.add_edge("C2", "a", "R", "y", "2026-01-01")
    assert len(m.timeline("C1")) == 1


def test_semantic_recall_ranks_relevant_first():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    m = CustomerMemory(conn, embedder=LocalEmbedder())
    m.init_tables()
    m.add_edge("C1", "customer", "COMPLAINED_ABOUT", "network coverage dropped calls", "2026-03-01")
    m.add_edge("C1", "customer", "ASKED_ABOUT", "international roaming rates", "2026-03-05")
    top = m.semantic_recall("C1", "signal keeps dropping", k=1)
    assert top[0].object.startswith("network coverage")


def test_consolidate_closes_prior_conflicting_edge():
    m = _mem()
    m.consolidate("C1", "customer", "PLAN_IS", "mobile_s", "2026-01-01")
    m.consolidate("C1", "customer", "PLAN_IS", "mobile_l", "2026-05-01")  # upgrade supersedes
    tl = m.timeline("C1")
    old = [e for e in tl if e.object == "mobile_s"][0]
    new = [e for e in tl if e.object == "mobile_l"][0]
    assert old.valid_to == "2026-05-01"  # closed by recency
    assert new.valid_to is None  # current
