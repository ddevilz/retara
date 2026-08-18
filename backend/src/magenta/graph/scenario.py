"""Single-customer graph runner for golden-scenario regression tests (Task 9.2).

`run_scenario` builds ONE real `Customer` + a matching `HiddenState` from a
scenario's kwargs, wires a GraphDeps that mixes REAL components (OfferCatalog
from configs/offers.yaml, ResponseOracle/SimParams, the actual compiled
StateGraph) with lightweight deterministic stand-ins for the ML brain / bandit
/ chat (mirroring the FakeRisk/FakeUplift/FakeBandit pattern already used in
tests/graph/conftest.py), runs the graph end to end on an in-memory sqlite
conn, and reports the disposition as a plain dict.

Anti-circularity note: hidden_kwargs (persuadable_segment/competitor_pull/etc.)
NEVER reaches the graph. It only parameterizes the fake risk/uplift stand-ins
built here, in test-harness code OUTSIDE the graph -- exactly like the existing
graph-node fakes. `sense()` still only ever calls `deps.risk.score(customer)` /
`deps.uplift.tau(customer)`; no node reads HiddenState.

The chat stub is a small rule-based "what would a sane diagnosis say" function
over OBSERVABLE customer fields only (same whitelist as nodes._OBSERVABLE_FIELDS)
-- deterministic per scenario, no LLM/network call, matching the "stub chat"
constraint for these tests.
"""
from __future__ import annotations

import uuid

from magenta.brain.risk import Band, Driver, RiskAssessment
from magenta.config import configs_dir
from magenta.db import get_conn
from magenta.graph.build import GraphDeps, build_graph
from magenta.graph.state import Diagnosis
from magenta.offers import Arm, OfferCatalog
from magenta.sim.oracle import ResponseOracle, SimParams
from magenta.sim.population import Customer, HiddenState, Segment

_CUSTOMER_ID = "GOLDEN-CUST"
_CAMPAIGN_ID = "GOLDEN-CAMPAIGN"
_ORACLE_SEED = 0

# ---- default (baseline "average" persona) Customer fields -----------------
# Only the fields a given scenario cares about are overridden via
# customer_kwargs; everything else falls back to this deterministic baseline.
# Values are picked so that, absent an override, NO arm-specific diagnosis
# rule below fires and every arm stays catalog-eligible (see
# magenta.offers.OfferCatalog.eligible) -- i.e. a "generic, nothing-special"
# customer.
_DEFAULT_CUSTOMER_KWARGS: dict = dict(
    tenure_months=24,
    contract="ONE_YEAR",
    monthly_charge=65.0,
    total_charges=1560.0,
    plan="STANDARD",
    data_gb_used_p50=16.0,
    data_allowance_gb=20.0,
    overage_events_90d=0,
    dropped_calls_30d=1,
    support_tickets_90d=0,
    nps_last=7,
    late_payments_12m=0,
    device_age_months=12,
    contract_end_days=60,
    gross_margin_monthly=40.0,
    clv_estimate=1200.0,
    senior_citizen=False,
    has_partner=True,
    has_dependents=False,
)

# ---- default hidden-state fields (only persuadable_segment usually set) ---
_DEFAULT_HIDDEN_KWARGS: dict = dict(
    theta_churn_base=0.0,
    theta_price_sens=0.0,
    persuadable_segment=Segment.PERSUADABLE,
    competitor_pull=0.1,
)

# segment -> (p_churn, tau) pair that classify_segment() maps back to that
# exact segment (thresholds mirror magenta.brain.uplift.classify_segment's
# defaults: risk_floor=0.25, tau_min=0.02). This is the "ground truth" stand-in
# for what a well-calibrated risk/uplift model would predict for a customer
# known (by construction, in this test harness) to belong to that segment.
_SEGMENT_RISK_UPLIFT: dict[Segment, tuple[float, float]] = {
    Segment.PERSUADABLE: (0.60, 0.20),
    Segment.SURE_THING: (0.10, 0.20),
    Segment.LOST_CAUSE: (0.60, 0.0),
    Segment.SLEEPING_DOG: (0.60, -0.20),
}

_STUB_DRIVERS = [
    Driver(feature="OVERAGE_EVENTS_90D", label="data overage events (90d)",
           shap_value=0.20, direction="UP"),
    Driver(feature="TENURE_MONTHS", label="tenure (months)",
           shap_value=-0.10, direction="DOWN"),
]


def _band_for(p: float) -> Band:
    # mirrors magenta.brain.risk._band_for (private, not exported)
    if p < 0.25:
        return Band.LOW
    if p < 0.50:
        return Band.MEDIUM
    if p < 0.75:
        return Band.HIGH
    return Band.CRITICAL


class _ScenarioRisk:
    """Deterministic risk stand-in: returns the p_churn baked in at build time."""

    def __init__(self, p_churn: float):
        self._p = p_churn

    def score(self, c: Customer) -> RiskAssessment:
        return RiskAssessment(p_churn=self._p, band=_band_for(self._p), drivers=_STUB_DRIVERS)


class _ScenarioUplift:
    """Deterministic uplift stand-in: returns the tau baked in at build time."""

    def __init__(self, tau: float):
        self._tau = tau

    def tau(self, c: Customer) -> float:
        return self._tau


class _FirstEligibleBandit:
    """Deterministic bandit stand-in: always picks the first (catalog ∩
    diagnosis)-eligible arm, in Arm-enum order. No randomness -> the same
    scenario always resolves to the same arm, which is what a regression
    pin needs.
    """

    def __init__(self):
        self.updates: list[tuple] = []

    def select(self, x, eligible: list[Arm]) -> tuple[Arm, float]:
        return eligible[0], 1.0

    def update(self, x, arm: Arm, reward: float) -> None:
        self.updates.append((arm, reward))

    def save(self, conn) -> None:
        pass


class _ScenarioParams:
    """Mirrors magenta.cli._GraphParams -- kept independent so this module
    doesn't have to import the (heavy) cli module for four constants."""

    freq_cap_days = 14
    freq_cap_max = 1
    value_cap = 40.0
    p90_clv = 2000.0


def _diagnose_stub(customer: Customer) -> Diagnosis:
    """Deterministic, rule-based stand-in for the diagnose LLM call.

    Looks only at observable customer fields (never hidden state). Each rule
    below picks the arm(s) that plausibly address the dominant signal; a
    generic broad menu is returned when nothing stands out. This is the
    "deterministic diagnosis per scenario" the brief calls for -- deterministic
    because it is a pure function of the observable customer, not an LLM call.
    """
    if customer.plan not in ("BASIC", "STANDARD", "PREMIUM"):
        return Diagnosis(
            root_cause_tags=["COMPETITOR_OFFER"],
            narrative="Multi-service bundle customer; a sticky add-on may help retain.",
            eligible_offer_ids=[Arm.BUNDLE_ADDON.value, Arm.ACKNOWLEDGE_AND_FIX.value],
            confidence=0.75,
        )
    if customer.device_age_months >= 24:
        return Diagnosis(
            root_cause_tags=["CONTRACT_EXPIRY"],
            narrative="Ageing device; an upgrade voucher may help retain.",
            eligible_offer_ids=[Arm.DEVICE_UPGRADE.value, Arm.ACKNOWLEDGE_AND_FIX.value],
            confidence=0.75,
        )
    if customer.dropped_calls_30d >= 5:
        return Diagnosis(
            root_cause_tags=["DROPPED_CALLS"],
            narrative="Frequent dropped calls; a network/service fix is indicated.",
            eligible_offer_ids=[Arm.NETWORK_PRIORITY_FIX.value, Arm.ACKNOWLEDGE_AND_FIX.value],
            confidence=0.8,
        )
    if customer.monthly_charge >= 85.0 and customer.overage_events_90d >= 1:
        return Diagnosis(
            root_cause_tags=["BILL_SHOCK"],
            narrative="High bill compounded by overage events; a credit or fix is indicated.",
            eligible_offer_ids=[Arm.BILL_CREDIT.value, Arm.ACKNOWLEDGE_AND_FIX.value],
            confidence=0.8,
        )
    if customer.overage_events_90d >= 3:
        return Diagnosis(
            root_cause_tags=["OVERAGE"],
            narrative="Recurring data overage; a boost or a better-fitting plan is indicated.",
            eligible_offer_ids=[Arm.DATA_BOOST.value, Arm.PLAN_DOWNSELL.value,
                                Arm.ACKNOWLEDGE_AND_FIX.value],
            confidence=0.75,
        )
    return Diagnosis(
        root_cause_tags=["GENERAL"],
        narrative="No single dominant signal; offering the broad menu.",
        eligible_offer_ids=[a.value for a in Arm if a is not Arm.NO_ACTION],
        confidence=0.5,
    )


class _ScenarioChat:
    """chat_structured stand-in bound to the one scenario customer. Ignores
    the prompt messages (no LLM/network call) and returns the rule-based
    diagnosis for that customer -- deterministic, per the brief."""

    def __init__(self, customer: Customer):
        self._customer = customer

    def chat_structured(self, role, messages, model_cls):
        return _diagnose_stub(self._customer)

    def chat(self, role, messages, **kw):
        return "stub"


def _build_customer(customer_id: str, overrides: dict) -> Customer:
    kwargs = {**_DEFAULT_CUSTOMER_KWARGS, **overrides}
    return Customer(customer_id=customer_id, **kwargs)


def _build_hidden(customer_id: str, overrides: dict) -> dict[str, HiddenState]:
    kwargs = {**_DEFAULT_HIDDEN_KWARGS, **overrides}
    return {customer_id: HiddenState(**kwargs)}


def run_scenario(customer_kwargs: dict, hidden_kwargs: dict, holdout: bool = False) -> dict:
    """Build one customer+hidden state, run the REAL compiled graph on a real
    Postgres conn, and report the disposition.

    Returns a dict with keys `contacted` (bool), `arm` (Arm | None), and
    `fulfilled` (bool) -- exactly what `magenta.evalx.golden._evaluate` checks
    against a scenario's `ExpectedDisposition`.

    `holdout` forces `OverallState["holdout"]` for the run: golden.py sets it
    from `scenario.expected.must_not_fulfill`, since state["holdout"] is the
    graph's only mechanism (Lab 7) for guaranteeing shadow-only, never-fulfilled
    behavior regardless of which arm decide() would otherwise pick.

    Tenant isolation replaces the old fresh-in-memory-db-per-call isolation:
    a brand new in-process db per call meant no scenario could ever see
    another scenario's (or another test run's) GUARDRAIL_CONTACTS/FULFILLMENTS
    rows -- e.g. the frequency cap (deps.params.freq_cap_max=1 per 14 days)
    would otherwise block every scenario after the first, since every golden
    scenario shares the same _CUSTOMER_ID. A shared committed Postgres conn
    has no such reset, so each call gets its own throwaway TENANT_ID instead;
    every tenant-scoped read/write in graph/tables.py is scoped to it, so
    scenarios (and repeated test runs) stay as isolated as the old in-memory
    db was, without needing a rollback.
    """
    customer = _build_customer(_CUSTOMER_ID, customer_kwargs)
    hidden = _build_hidden(_CUSTOMER_ID, hidden_kwargs)
    segment = hidden[_CUSTOMER_ID].persuadable_segment
    p_churn, tau = _SEGMENT_RISK_UPLIFT[segment]
    tenant_id = f"scenario_{uuid.uuid4().hex}"

    with get_conn() as conn:
        sim_params = SimParams.load(configs_dir() / "sim_params.yaml")
        deps = GraphDeps(
            risk=_ScenarioRisk(p_churn),
            uplift=_ScenarioUplift(tau),
            bandit=_FirstEligibleBandit(),
            catalog=OfferCatalog.load(configs_dir() / "offers.yaml"),
            oracle=ResponseOracle(hidden, params=sim_params, seed=_ORACLE_SEED),
            conn=conn,
            params=_ScenarioParams(),
            chat=_ScenarioChat(customer),
            load_customer=lambda cid: customer,
            campaign_id=_CAMPAIGN_ID,
            tenant_id=tenant_id,
        )
        graph = build_graph(deps)
        init_state = {
            "customer_id": customer.customer_id, "campaign_id": _CAMPAIGN_ID,
            "consent_flags": {"MARKETING": True},
            "risk": None, "diagnosis": None, "offer": None, "verdict": None,
            "fulfillment": None, "outcome": None, "messages": [], "audit_log": [],
            "requires_approval": False, "holdout": holdout,
        }
        final = graph.invoke(
            init_state,
            config={"configurable": {"thread_id": f"{deps.tenant_id}:{customer.customer_id}:{_CAMPAIGN_ID}"}},
        )

    risk_report = final.get("risk")
    contacted = bool(risk_report is not None and risk_report.engage)
    offer = final.get("offer")
    arm = offer.arm if offer is not None else None
    fulfillment = final.get("fulfillment") or {}
    status = fulfillment.get("STATUS") or fulfillment.get("status")
    fulfilled = status == "FULFILLED"
    return {"contacted": contacted, "arm": arm, "fulfilled": fulfilled}
