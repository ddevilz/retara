"""Assemble GraphDeps for the API process.

Checked labs 0-9 for a reusable factory first: `magenta.graph`'s public
surface (`magenta.graph.__init__`) exports only `GraphDeps`/`act`/`diagnose`/
`sense`/`run_scenario` — no `default_deps()` exists anywhere in the package
(grepped the full source tree). So there is no "preferred path" to try; this
module mirrors the CLI `run-one` wiring exactly (`magenta/cli.py::run_one`)
instead of guessing at a factory that isn't there. Doing that probe as a
function-level `try: from magenta.graph import default_deps` would also
violate this repo's hard "no function-level imports" rule (CLAUDE.md) for a
branch that can never succeed today, so it's left out rather than special-cased.

`get_graph_deps()` is cached (@lru_cache) so the risk/uplift models load once
per process, not once per request. `find_customer()` shares one cached
population (same DEMO_POP_N/SEED as `magenta.api.data_access`, imported from
there rather than re-declared, so the two can never drift apart) with
`GraphDeps.load_customer`, so any customer_id the /api/customers list serves
also resolves for the graph.
"""
from __future__ import annotations

from functools import lru_cache

from magenta.api.data_access import DEMO_POP_N, DEMO_POP_SEED
from magenta.brain.bandit import ThompsonBandit
from magenta.brain.features import FEATURE_NAMES
from magenta.brain.risk import RiskModel
from magenta.brain.training import build_training_data
from magenta.brain.uplift import UpliftModel
from magenta.config import configs_dir
from magenta.db import get_conn
from magenta.graph.build import GraphDeps
from magenta.graph.tables import DEFAULT_TENANT_ID
from magenta.llm import chat, chat_structured
from magenta.offers import Arm, OfferCatalog
from magenta.sim.oracle import ResponseOracle, SimParams
from magenta.sim.population import Customer, generate_population


class _GraphParams:
    """Mirrors cli.py's `_GraphParams` — the freq-cap/value-cap knobs the
    guardrail node reads off `deps.params`."""

    freq_cap_days = 14
    freq_cap_max = 1
    value_cap = 40.0
    p90_clv = 2000.0


class _ChatShim:
    """Adapts module-level llm.chat/chat_structured to the deps.chat interface
    (same shim cli.py uses for `run-one`/`chat`/`ablation`)."""

    def chat(self, role, messages, **kw):
        return chat(role, messages, **kw)

    def chat_structured(self, role, messages, model_cls):
        return chat_structured(role, messages, model_cls)


def _load_or_train_risk(seed: int) -> RiskModel:
    """Mirrors cli.py::_load_or_train_risk. `RiskModel.load()` with no
    argument resolves the data_dir()-anchored default path (NOT the brief's
    cwd-relative "data/risk.pkl" literal, which only resolves if the process
    cwd happens to be the repo root)."""
    try:
        return RiskModel.load()
    except FileNotFoundError:
        td = build_training_data(n=3000, seed=seed)
        model = RiskModel().fit(td.customers, td.churned)
        model.save()
        return model


def _load_or_train_uplift(seed: int) -> UpliftModel:
    try:
        return UpliftModel.load()
    except FileNotFoundError:
        td = build_training_data(n=3000, seed=seed)
        model = UpliftModel().fit(td.customers, td.treated, td.retained)
        model.save()
        return model


@lru_cache(maxsize=1)
def _demo_customers() -> dict[str, Customer]:
    """One generation of the same demo population `magenta.api.data_access`
    serves `/api/customers` from, keyed by id for O(1) lookup. `generate_population`
    returns `(list[Customer], HiddenStore)` — the hidden half is discarded
    immediately here and never touched again."""
    customers, _hidden = generate_population(DEMO_POP_N, DEMO_POP_SEED)
    return {c.customer_id: c for c in customers}


def find_customer(customer_id: str) -> Customer | None:
    return _demo_customers().get(customer_id)


@lru_cache(maxsize=1)
def get_graph_deps() -> GraphDeps:
    """Real GraphDeps for the API process — same risk/uplift/bandit/catalog/
    oracle wiring `magenta run-one` uses, but with a `load_customer` that
    resolves ANY demo-population id (the CLI binds a single customer per
    invocation via a closure; the API's deps are a long-lived singleton
    shared across requests for different customers, so that closure shape
    doesn't fit here)."""
    seed = DEMO_POP_SEED
    conn = get_conn()
    _, hidden = generate_population(DEMO_POP_N, seed=seed)
    bandit = ThompsonBandit(dim=len(FEATURE_NAMES), arms=list(Arm), seed=seed)
    # DEFAULT_TENANT_ID: no per-tenant get_graph_deps(tenant_id) yet -- Phase 1.3
    # replaces this with the tenant_id argument that call gets.
    bandit.load(conn, DEFAULT_TENANT_ID)  # no-op prior if no rows yet
    sim_params = SimParams.load(configs_dir() / "sim_params.yaml")
    return GraphDeps(
        risk=_load_or_train_risk(seed),
        uplift=_load_or_train_uplift(seed),
        bandit=bandit,
        catalog=OfferCatalog.load(configs_dir() / "offers.yaml"),
        oracle=ResponseOracle(hidden, params=sim_params, seed=seed),
        conn=conn,
        params=_GraphParams(),
        chat=_ChatShim(),
        load_customer=find_customer,
    )
