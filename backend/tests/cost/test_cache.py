from magenta.cost.cache import SemanticCache
from magenta.memory.embed import LocalEmbedder


def _cache(conn):
    return SemanticCache(conn, LocalEmbedder())


def test_near_duplicate_hits_exact_miss_is_none(db_conn):
    c = _cache(db_conn)
    assert c.get("bill shock, price sensitive, contract ending") is None      # miss
    c.put("bill shock, price sensitive, contract ending", "PLAN_DOWNSELL")
    assert c.get("bill shock, price sensitive, contract ending") == "PLAN_DOWNSELL"  # exact
    assert c.get("customer has bill shock and is price sensitive, contract about to end") == "PLAN_DOWNSELL"  # near-dup
    assert c.get("the weather is sunny today") is None                        # unrelated -> miss
