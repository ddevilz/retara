"""Per-tenant GraphDeps.

Every singleton this module used to hold is gone: one population, one risk model, one
uplift model, one bandit posterior, all shared by every caller. Each is now keyed by
tenant and held in a bounded, expiring cache.

Models are never trained here. `_load_or_train_risk` used to train LightGBM inline when
no artifact existed, which for a new tenant meant a multi-minute HTTP request. A missing
artifact is now `ModelsNotReady` -> 503, and provisioning is somebody else's job:
`magenta tenant provision` today, a background job in Phase 1.4.
"""
from __future__ import annotations

from magenta.api.population import get_population
from magenta.brain.bandit import ThompsonBandit
from magenta.brain.features import FEATURE_NAMES
from magenta.brain.risk import RiskModel
from magenta.brain.uplift import UpliftModel
from magenta.config import configs_dir
from magenta.db import get_conn
from magenta.graph.build import GraphDeps
from magenta.llm import chat, chat_structured
from magenta.offers import Arm, OfferCatalog
from magenta.sim.oracle import ResponseOracle, SimParams
from magenta.storage import risk_model_path, uplift_model_path
from magenta.tenancy import BoundedTTLCache, tenant_seed

# Each entry holds two LightGBM model sets, a bandit posterior and a population.
# maxsize is the memory bound; ttl_seconds bounds staleness after the Phase 1.4 worker
# retrains in a different process and cannot invalidate this one. on_evict closes the
# Connection get_graph_deps() checked out for this entry -- without it, every eviction
# (TTL expiry, LRU, overwrite, invalidate, clear) leaks a pooled connection forever and
# the pool (~15 conns) exhausts after enough tenant rotation.
DEPS_CACHE = BoundedTTLCache(maxsize=8, ttl_seconds=900, on_evict=lambda deps: deps.conn.close())


class ModelsNotReady(Exception):
    """This tenant has no trained artifacts yet. Provision it, then retry."""


class _GraphParams:
    freq_cap_days = 14
    freq_cap_max = 1
    value_cap = 40.0
    p90_clv = 2000.0


class _ChatShim:
    def chat(self, role, messages, **kw):
        return chat(role, messages, **kw)

    def chat_structured(self, role, messages, model_cls):
        return chat_structured(role, messages, model_cls)


def get_graph_deps(tenant_id: str) -> GraphDeps:
    cached = DEPS_CACHE.get(tenant_id)
    if cached is not None:
        return cached

    try:
        risk = RiskModel.load(risk_model_path(tenant_id))
        uplift = UpliftModel.load(uplift_model_path(tenant_id))
    except FileNotFoundError as exc:
        raise ModelsNotReady(f"no trained models for tenant {tenant_id}") from exc

    seed = tenant_seed(tenant_id)
    pop = get_population(tenant_id)

    conn = get_conn()
    bandit = ThompsonBandit(dim=len(FEATURE_NAMES), arms=list(Arm), seed=seed)
    bandit.load(conn, tenant_id)

    deps = GraphDeps(
        risk=risk,
        uplift=uplift,
        bandit=bandit,
        catalog=OfferCatalog.load(configs_dir() / "offers.yaml"),
        oracle=ResponseOracle(
            pop.hidden,
            params=SimParams.load(configs_dir() / "sim_params.yaml"),
            seed=seed,
        ),
        conn=conn,
        params=_GraphParams(),
        chat=_ChatShim(),
        load_customer=pop.customers.get,
        tenant_id=tenant_id,
    )
    DEPS_CACHE.put(tenant_id, deps)
    return deps
