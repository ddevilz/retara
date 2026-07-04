from magenta.brain.bandit import ThompsonBandit
from magenta.brain.features import FEATURE_NAMES
from magenta.brain.policy import BrainPolicy
from magenta.brain.risk import RiskModel
from magenta.brain.training import build_training_data
from magenta.brain.uplift import UpliftModel
from magenta.config import configs_dir
from magenta.offers import Arm, OfferCatalog


def _policy(budget=None):
    td = build_training_data(n=3000, seed=41)
    rm = RiskModel().fit(td.customers, td.churned)
    um = UpliftModel().fit(td.customers, td.treated, td.retained)
    cat = OfferCatalog.load(configs_dir() / "offers.yaml")
    bandit = ThompsonBandit(dim=len(FEATURE_NAMES), arms=list(Arm))
    return BrainPolicy(rm, um, bandit, cat, budget=budget), td


def test_decide_returns_offer_or_none():
    pol, td = _policy()
    out = pol.decide(td.customers[0])
    assert out is None or hasattr(out, "arm")


def test_budget_never_exceeded():
    pol, td = _policy(budget=50.0)
    pol.reset_budget()
    spent = 0.0
    # feed cohort sorted the way run_experiment will (policy handles its own gating).
    for c in td.customers[:500]:
        out = pol.decide(c)
        if out is not None and out.arm != Arm.NO_ACTION:
            spent += out.cost
    assert spent <= 50.0 + 1e-9


def test_reset_budget_restores_capacity():
    pol, td = _policy(budget=10.0)
    pol.reset_budget()
    for c in td.customers[:500]:
        pol.decide(c)
    pol.reset_budget()
    # after reset at least one offer becomes possible again for a persuadable
    assert pol._remaining == 10.0


def test_risk_floor_constants_stay_equal():
    """BrainPolicy._RISK_FLOOR must match classify_segment's default risk_floor —
    the bandit-episodes CLI's inline gate equivalence depends on it."""
    import inspect

    from magenta.brain.policy import _RISK_FLOOR
    from magenta.brain.uplift import classify_segment

    default_floor = inspect.signature(classify_segment).parameters["risk_floor"].default
    assert _RISK_FLOOR == default_floor
