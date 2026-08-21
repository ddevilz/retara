import pytest

from magenta.api.deps import DEPS_CACHE, ModelsNotReady, get_graph_deps
from tests.db_fixtures import TENANT_A, TENANT_B


def setup_function():
    DEPS_CACHE.clear()


def test_missing_models_raise_not_ready(monkeypatch, tmp_path):
    """A cold tenant must fail fast, not train LightGBM inside a request."""
    monkeypatch.setenv("MAGENTA_MODEL_DIR", str(tmp_path))
    with pytest.raises(ModelsNotReady):
        get_graph_deps("org_never_provisioned")


def test_deps_are_cached_per_tenant(provisioned_tenants):
    assert get_graph_deps(TENANT_A) is get_graph_deps(TENANT_A)


def test_tenants_get_distinct_deps(provisioned_tenants):
    assert get_graph_deps(TENANT_A) is not get_graph_deps(TENANT_B)


def test_load_customer_is_tenant_scoped(provisioned_tenants):
    """Each tenant's GraphDeps.load_customer must resolve through THAT tenant's own
    population, not a shared/global one.

    Deviation from the brief's literal assertion (`b.load_customer(a_id) is None`):
    customer_id is index-based ("C0000000", ...), not seed-based, so the same id
    exists in every tenant's population by design (see magenta.sim.population and
    tests/api/test_run_one_tenant_scoping.py, which relies on this to reuse one
    customer_id across two tenants). Isolation means tenant B resolves ITS OWN
    Customer object for that id, with its own attribute values -- not that the id
    fails to resolve at all. The brief's `is None` assertion is unsatisfiable under
    the repo's own established id-space design and would have been a false failure.
    """
    from magenta.api.population import get_population

    a_id = next(iter(get_population(TENANT_A).customers))
    a = get_graph_deps(TENANT_A)
    b = get_graph_deps(TENANT_B)

    assert a.load_customer(a_id) is get_population(TENANT_A).customers[a_id]
    assert b.load_customer(a_id) is get_population(TENANT_B).customers[a_id]
    assert a.load_customer(a_id) is not b.load_customer(a_id), (
        "tenant B's lookup returned tenant A's Customer object"
    )


def test_oracle_uses_this_tenants_hidden_store(provisioned_tenants):
    """The oracle must be built from the tenant's own HiddenStore, not a shared one."""
    from magenta.api.population import get_population

    assert get_graph_deps(TENANT_A).oracle.hidden is get_population(TENANT_A).hidden


def test_cache_is_bounded(provisioned_tenants):
    """maxsize is the OOM guard — each entry holds two LightGBM model sets."""
    assert DEPS_CACHE._maxsize <= 16
