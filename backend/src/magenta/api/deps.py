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
population (own DEMO_POP_N/SEED, defined below) with `GraphDeps.load_customer`,
so any customer_id the /api/customers list serves also resolves for the graph.

Phase 1.3 Task 3 note: `DEMO_POP_N`/`DEMO_POP_SEED` used to be imported from
`magenta.api.data_access`, which owned the one shared demo-population seed.
Task 3 deleted that module's population entirely (collapsed into
`magenta.api.population`, per-tenant) — this module is untouched by Task 3
per its own brief ("Task 4 replaces `_demo_customers`/`find_customer`"), so
the constants are inlined here, same values, so this file keeps importing
and behaving exactly as before until Task 4 rewires it onto
`magenta.api.population.get_population(tenant_id)`.
"""
from __future__ import annotations

from functools import lru_cache

from magenta.brain.bandit import ThompsonBandit
from magenta.brain.features import FEATURE_NAMES
from magenta.brain.risk import RiskModel
from magenta.brain.training import build_training_data
from magenta.brain.uplift import UpliftModel
from magenta.config import configs_dir, data_dir
from magenta.db import get_conn
from magenta.graph.build import GraphDeps
from magenta.graph.tables import DEFAULT_TENANT_ID
from magenta.llm import chat, chat_structured
from magenta.offers import Arm, OfferCatalog
from magenta.sim.oracle import ResponseOracle, SimParams
from magenta.sim.population import Customer, generate_population


# Population seed/size used for the demo customer directory. Must match the
# seed the graph/experiment use so IDs line up with AUDIT_LOG rows. Formerly
# imported from magenta.api.data_access (Task 3 deleted that copy — see
# module docstring).
DEMO_POP_N = 2000
DEMO_POP_SEED = 7


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
    """Mirrors cli.py::_load_or_train_risk. Single-tenant demo default
    (Phase 1.3 pre-Task-4): `get_graph_deps()` below is still a process-wide
    singleton keyed on DEFAULT_TENANT_ID, so this keeps the old shared
    data_dir()-anchored path as a literal here rather than inventing a new
    default-path abstraction. Task 4 replaces this with a tenant path via
    magenta.storage."""
    path = data_dir() / "models" / "risk.joblib"
    try:
        return RiskModel.load(path)
    except FileNotFoundError:
        td = build_training_data(n=3000, seed=seed)
        model = RiskModel().fit(td.customers, td.churned)
        model.save(path)
        return model


def _load_or_train_uplift(seed: int) -> UpliftModel:
    """Single-tenant demo default (Phase 1.3 pre-Task-4); see _load_or_train_risk."""
    path = data_dir() / "models" / "uplift.joblib"
    try:
        return UpliftModel.load(path)
    except FileNotFoundError:
        td = build_training_data(n=3000, seed=seed)
        model = UpliftModel().fit(td.customers, td.treated, td.retained)
        model.save(path)
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
