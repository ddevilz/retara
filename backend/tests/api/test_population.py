from magenta.api.population import (
    POPULATION_CACHE,
    get_customer,
    get_population,
    list_customers,
)
from tests.db_fixtures import TENANT_A, TENANT_B


def setup_function():
    POPULATION_CACHE.clear()


def test_population_is_deterministic_per_tenant():
    first = get_population(TENANT_A)
    POPULATION_CACHE.clear()
    second = get_population(TENANT_A)
    assert [c.customer_id for c in first.summaries[:20]] == \
           [c.customer_id for c in second.summaries[:20]]


def test_tenants_get_different_populations():
    # customer_id is index-based, not seeded (see test_list_and_get_are_tenant_scoped),
    # so both tenants share the same id space by design -- what must differ is the
    # simulated data itself (proving tenant_seed actually drives generate_population).
    a = get_population(TENANT_A)
    b = get_population(TENANT_B)
    assert a.summaries != b.summaries, "two tenants received the identical simulated population"


def test_population_is_cached():
    assert get_population(TENANT_A) is get_population(TENANT_A)


def test_list_and_get_are_tenant_scoped():
    # customer_id is index-based (`f"C{i:07d}"` in magenta.sim.population.generate_population)
    # and NOT seeded, so every tenant's sandbox reuses the same id space by design --
    # tests/api/test_tenant_isolation.py::test_customer_360_audit_is_tenant_scoped
    # already pins a 200 (not 404) for tenant B looking up tenant A's "C0000000". What
    # must be tenant-scoped is the *data* behind that id: each tenant's simulated
    # attributes come from its own get_population(tenant_id), not a shared cache entry.
    a_first = list_customers(TENANT_A, limit=1)[0]
    b_view = get_customer(TENANT_B, a_first.customer_id)
    assert b_view is not None
    assert b_view != a_first, "tenant B's lookup returned tenant A's cached record"


def test_hidden_state_never_reaches_summaries():
    """Anti-circularity: HiddenStore is simulator-private and must not surface on the
    objects the API serves."""
    forbidden = {"theta_churn_base", "theta_price_sens",
                 "persuadable_segment", "competitor_pull"}
    summary = list_customers(TENANT_A, limit=1)[0]
    assert forbidden.isdisjoint(summary.model_dump().keys())
