from magenta.cost.cache import SemanticCache
from magenta.memory.embed import LocalEmbedder
from tests.db_fixtures import TENANT_A, TENANT_B


def _cache(conn, tenant_id=TENANT_A):
    return SemanticCache(conn, tenant_id, LocalEmbedder())


def test_near_duplicate_hits_exact_miss_is_none(db_conn):
    c = _cache(db_conn)
    assert c.get("bill shock, price sensitive, contract ending") is None      # miss
    c.put("bill shock, price sensitive, contract ending", "PLAN_DOWNSELL")
    assert c.get("bill shock, price sensitive, contract ending") == "PLAN_DOWNSELL"  # exact
    assert c.get("customer has bill shock and is price sensitive, contract about to end") == "PLAN_DOWNSELL"  # near-dup
    assert c.get("the weather is sunny today") is None                        # unrelated -> miss


def test_cache_never_serves_across_tenants(db_conn, _shared_embedder):
    """A cross-tenant hit would leak one tenant's LLM output to another."""
    a = SemanticCache(db_conn, TENANT_A, _shared_embedder)
    b = SemanticCache(db_conn, TENANT_B, _shared_embedder)
    a.put("why is my bill so high", "TENANT A PRIVATE ANSWER")

    assert a.get("why is my bill so high") == "TENANT A PRIVATE ANSWER"
    assert b.get("why is my bill so high") is None
