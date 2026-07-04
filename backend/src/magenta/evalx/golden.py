"""Golden scenarios: pinned dispositions for the retention agent (Task 9.2).

Regression pins for BEHAVIOR, not implementation: each `GoldenScenario` names a
customer/hidden-state combo and the disposition the agent MUST (or must not)
produce. `run_golden()` runs every scenario through the REAL compiled graph
(`magenta.graph.run_scenario`) with a deterministic, rule-based chat stub (no
LLM/network call) and reports pass/fail per scenario, so a future change that
silently breaks one of these must-hold behaviors fails loudly here instead of
being caught (or missed) downstream.

`_evaluate` is the patchable seam: `run_golden()` calls it by module-level
name, so tests can stub it out entirely to verify the harness structure
without running the real graph (see tests/evalx/test_golden.py).
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from magenta.graph import run_scenario
from magenta.offers import Arm

_DISCOUNT_ARMS = [Arm.BILL_CREDIT, Arm.PLAN_DOWNSELL]


class ExpectedDisposition(BaseModel):
    should_contact: bool = True
    allowed_arms: list[Arm] | None = None
    forbidden_arms: list[Arm] = Field(default_factory=list)
    must_not_fulfill: bool = False  # holdout: agent must decide, but never act


class GoldenScenario(BaseModel):
    name: str
    customer_kwargs: dict = Field(default_factory=dict)
    hidden_kwargs: dict = Field(default_factory=dict)
    expected: ExpectedDisposition


class GoldenResult(BaseModel):
    name: str
    passed: bool
    detail: str


GOLDEN_SCENARIOS: list[GoldenScenario] = [
    GoldenScenario(
        name="sleeping_dog_no_contact",
        customer_kwargs={"tenure_months": 60, "nps_last": 9},
        hidden_kwargs={"persuadable_segment": "SLEEPING_DOG"},
        expected=ExpectedDisposition(should_contact=False),
    ),
    GoldenScenario(
        name="bill_shock_gets_ack_or_credit",
        customer_kwargs={"overage_events_90d": 5, "monthly_charge": 90.0},
        hidden_kwargs={"persuadable_segment": "PERSUADABLE"},
        expected=ExpectedDisposition(
            allowed_arms=[Arm.ACKNOWLEDGE_AND_FIX, Arm.BILL_CREDIT]),
    ),
    GoldenScenario(
        name="network_complainer_not_offered_discount",
        customer_kwargs={"dropped_calls_30d": 12},
        hidden_kwargs={"persuadable_segment": "PERSUADABLE"},
        expected=ExpectedDisposition(
            allowed_arms=[Arm.NETWORK_PRIORITY_FIX, Arm.ACKNOWLEDGE_AND_FIX],
            forbidden_arms=_DISCOUNT_ARMS),
    ),
    GoldenScenario(
        name="holdout_never_fulfilled",
        customer_kwargs={"overage_events_90d": 4},
        hidden_kwargs={"persuadable_segment": "PERSUADABLE"},
        expected=ExpectedDisposition(must_not_fulfill=True),
    ),
    GoldenScenario(
        name="lost_cause_no_expensive_offer",
        customer_kwargs={"contract_end_days": 3},
        hidden_kwargs={"persuadable_segment": "LOST_CAUSE"},
        expected=ExpectedDisposition(should_contact=False,
                                     forbidden_arms=[Arm.DEVICE_UPGRADE]),
    ),
    GoldenScenario(
        name="sure_thing_not_wasted",
        customer_kwargs={"tenure_months": 48, "nps_last": 8},
        hidden_kwargs={"persuadable_segment": "SURE_THING"},
        expected=ExpectedDisposition(should_contact=False),
    ),
    GoldenScenario(
        name="device_ageing_upgrade_eligible",
        customer_kwargs={"device_age_months": 40},
        hidden_kwargs={"persuadable_segment": "PERSUADABLE"},
        expected=ExpectedDisposition(allowed_arms=[Arm.DEVICE_UPGRADE,
                                                   Arm.ACKNOWLEDGE_AND_FIX]),
    ),
    GoldenScenario(
        name="high_data_user_gets_boost_or_downsell",
        customer_kwargs={"overage_events_90d": 8},
        hidden_kwargs={"persuadable_segment": "PERSUADABLE"},
        expected=ExpectedDisposition(allowed_arms=[Arm.DATA_BOOST, Arm.PLAN_DOWNSELL,
                                                   Arm.ACKNOWLEDGE_AND_FIX]),
    ),
    GoldenScenario(
        name="competitor_pull_within_authority",
        customer_kwargs={"monthly_charge": 70.0},
        hidden_kwargs={"persuadable_segment": "PERSUADABLE", "competitor_pull": 0.8},
        expected=ExpectedDisposition(),
    ),
    GoldenScenario(
        name="low_margin_offer_rejected",
        customer_kwargs={"gross_margin_monthly": 2.0},
        hidden_kwargs={"persuadable_segment": "PERSUADABLE"},
        expected=ExpectedDisposition(forbidden_arms=[Arm.DEVICE_UPGRADE]),
    ),
    GoldenScenario(
        name="bundle_addon_for_multiservice",
        customer_kwargs={"plan": "TRIPLE_PLAY"},
        hidden_kwargs={"persuadable_segment": "PERSUADABLE"},
        expected=ExpectedDisposition(allowed_arms=[Arm.BUNDLE_ADDON,
                                                   Arm.ACKNOWLEDGE_AND_FIX]),
    ),
]


def _evaluate(scenario: GoldenScenario) -> GoldenResult:
    """Run one scenario through the graph and check its disposition.

    Builds a single customer + hidden state from the scenario kwargs (via
    `magenta.graph.run_scenario`), runs the compiled graph on an in-memory
    conn with a deterministic chat stub, then checks: contact decision,
    chosen arm against allowed/forbidden, and (for holdout) that nothing was
    fulfilled.

    NOTE (brief deviation, documented): `must_not_fulfill` scenarios are run
    with `state["holdout"]=True` -- that flag is the graph's only mechanism
    (Lab 7) for guaranteeing shadow-only/never-fulfilled behavior regardless
    of which arm decide() would otherwise pick, so it is what this check
    actually needs to exercise.
    """
    exp = scenario.expected
    result = run_scenario(scenario.customer_kwargs, scenario.hidden_kwargs,
                          holdout=exp.must_not_fulfill)
    contacted = result.get("contacted", False)
    arm = result.get("arm")
    fulfilled = result.get("fulfilled", False)
    arm_label = arm.value if arm is not None else "None"

    if exp.should_contact is False and contacted:
        return GoldenResult(name=scenario.name, passed=False,
                            detail=f"should not contact but chose {arm_label}")
    if exp.must_not_fulfill and fulfilled:
        return GoldenResult(name=scenario.name, passed=False,
                            detail="holdout customer was fulfilled")
    if arm is not None and arm in exp.forbidden_arms:
        return GoldenResult(name=scenario.name, passed=False,
                            detail=f"forbidden arm {arm_label} chosen")
    if exp.allowed_arms is not None and contacted and arm not in exp.allowed_arms:
        return GoldenResult(name=scenario.name, passed=False,
                            detail=f"arm {arm_label} not in allowed {exp.allowed_arms}")
    return GoldenResult(name=scenario.name, passed=True, detail="ok")


def run_golden() -> list[GoldenResult]:
    return [_evaluate(s) for s in GOLDEN_SCENARIOS]
