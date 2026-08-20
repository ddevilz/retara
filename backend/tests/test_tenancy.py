import time

from magenta.tenancy import BoundedTTLCache, tenant_seed


def test_tenant_seed_is_deterministic():
    assert tenant_seed("org_abc") == tenant_seed("org_abc")


def test_tenant_seed_differs_per_tenant():
    assert tenant_seed("org_abc") != tenant_seed("org_xyz")


def test_tenant_seed_is_stable_across_processes():
    """Pinned literals, not a self-comparison: hash() is salted per process and would
    pass a self-comparison while silently changing between restarts.

    Values are int(sha256(tenant_id).hexdigest()[:8], 16), computed 2026-08-14.
    """
    assert tenant_seed("org_abc") == 230667760
    assert tenant_seed("org_xyz") == 3745499098


def test_cache_evicts_least_recently_used():
    c = BoundedTTLCache(maxsize=2, ttl_seconds=60)
    c.put("a", 1)
    c.put("b", 2)
    c.get("a")            # 'a' is now most-recently used, so 'b' is the victim
    c.put("c", 3)
    assert c.get("b") is None
    assert c.get("a") == 1
    assert c.get("c") == 3


def test_cache_expires_on_ttl():
    c = BoundedTTLCache(maxsize=8, ttl_seconds=0.05)
    c.put("a", 1)
    assert c.get("a") == 1
    time.sleep(0.06)
    assert c.get("a") is None


def test_invalidate_removes_one_key_only():
    c = BoundedTTLCache(maxsize=8, ttl_seconds=60)
    c.put("a", 1)
    c.put("b", 2)
    c.invalidate("a")
    assert c.get("a") is None
    assert c.get("b") == 2


def test_invalidate_unknown_key_is_not_an_error():
    BoundedTTLCache(maxsize=2, ttl_seconds=60).invalidate("nope")
