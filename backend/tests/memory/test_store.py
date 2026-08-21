import pytest
from sqlalchemy import text

from magenta.memory.embed import LocalEmbedder
from magenta.memory.store import CustomerMemory
from tests.db_fixtures import TENANT_A, TENANT_B

pytestmark = pytest.mark.slow


def _mem(conn):
    return CustomerMemory(conn, TENANT_A)


def test_add_and_timeline_ordered(db_conn):
    m = _mem(db_conn)
    m.add_edge("C1", "customer", "COMPLAINED_ABOUT", "coverage", "2026-03-01")
    m.add_edge("C1", "agent", "GAVE", "bill_credit", "2026-04-01")
    tl = m.timeline("C1")
    # valid_from is now a full ISO timestamp (Postgres widens the date-only
    # input to midnight); compare by date prefix, not exact string.
    assert [e.valid_from[:10] for e in tl] == ["2026-03-01", "2026-04-01"]  # temporal order
    assert tl[1].object == "bill_credit"


def test_timeline_is_per_customer(db_conn):
    m = _mem(db_conn)
    m.add_edge("C1", "a", "R", "x", "2026-01-01")
    m.add_edge("C2", "a", "R", "y", "2026-01-01")
    assert len(m.timeline("C1")) == 1


def test_semantic_recall_ranks_relevant_first(db_conn):
    m = CustomerMemory(db_conn, TENANT_A, embedder=LocalEmbedder())
    m.add_edge("C1", "customer", "COMPLAINED_ABOUT", "network coverage dropped calls", "2026-03-01")
    m.add_edge("C1", "customer", "ASKED_ABOUT", "international roaming rates", "2026-03-05")
    top = m.semantic_recall("C1", "signal keeps dropping", k=1)
    assert top[0].object.startswith("network coverage")


def test_consolidate_closes_prior_conflicting_edge(db_conn):
    m = _mem(db_conn)
    m.consolidate("C1", "customer", "PLAN_IS", "mobile_s", "2026-01-01")
    m.consolidate("C1", "customer", "PLAN_IS", "mobile_l", "2026-05-01")  # upgrade supersedes
    tl = m.timeline("C1")
    old = [e for e in tl if e.object == "mobile_s"][0]
    new = [e for e in tl if e.object == "mobile_l"][0]
    # valid_to is now a full ISO timestamp; compare by date prefix.
    assert old.valid_to[:10] == "2026-05-01"  # closed by recency
    assert new.valid_to is None  # current


def test_add_edge_returns_id_on_postgres(db_conn):
    """lastrowid is SQLite-only; Postgres needs INSERT ... RETURNING."""
    mem = CustomerMemory(db_conn, TENANT_A)
    edge_id = mem.add_edge("CUST_0001", "customer", "reported", "bill shock", "2026-01-01")
    assert isinstance(edge_id, int) and edge_id > 0


def test_timeline_is_tenant_isolated(db_conn):
    a = CustomerMemory(db_conn, TENANT_A)
    b = CustomerMemory(db_conn, TENANT_B)
    a.add_edge("CUST_0001", "customer", "reported", "bill shock", "2026-01-01")
    assert len(a.timeline("CUST_0001")) == 1
    assert b.timeline("CUST_0001") == [], "tenant B must not see tenant A's edges"


def test_embedding_roundtrip_preserves_float32(db_conn):
    """Embeddings are float32 while bandit posteriors are float64 -- neither blob
    records its dtype, so this pins the convention."""
    import numpy as np

    class FakeEmbedder:
        def encode(self, texts):
            return np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

    mem = CustomerMemory(db_conn, TENANT_A, embedder=FakeEmbedder())
    mem.add_edge("CUST_0001", "customer", "reported", "bill shock", "2026-01-01")
    hits = mem.semantic_recall("CUST_0001", "bill", k=1)
    assert len(hits) == 1

    raw = db_conn.execute(text(
        'SELECT "EMBEDDING" FROM "MEMORY_EDGES" WHERE "TENANT_ID" = :t'
    ), {"t": TENANT_A}).scalar_one()
    back = np.frombuffer(raw, dtype=np.float32)
    assert back.dtype == np.float32
    np.testing.assert_allclose(back, [0.1, 0.2, 0.3], rtol=1e-6)
    assert len(raw) == 3 * 4, "float32 is 4 bytes/element; a float64 write would be 24"


def test_consolidate_is_tenant_isolated(db_conn):
    """consolidate()'s open-edge lookup must not close/see another tenant's edge."""
    a = CustomerMemory(db_conn, TENANT_A)
    b = CustomerMemory(db_conn, TENANT_B)
    a.consolidate("C1", "customer", "PLAN_IS", "mobile_s", "2026-01-01")
    # Tenant B consolidating the "same" customer/subject/relation must not see
    # or close tenant A's open edge -- it should just insert its own.
    b.consolidate("C1", "customer", "PLAN_IS", "mobile_l", "2026-05-01")

    a_tl = a.timeline("C1")
    b_tl = b.timeline("C1")
    assert len(a_tl) == 1 and a_tl[0].valid_to is None, "tenant A edge must remain open"
    assert len(b_tl) == 1 and b_tl[0].object == "mobile_l"
