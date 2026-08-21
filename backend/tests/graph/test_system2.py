
from magenta.brain.risk import Band, Driver
from magenta.brain.uplift import Segment
from magenta.graph.state import Diagnosis, RiskUpliftReport, Timing
from magenta.graph.system2 import deliberate, should_deliberate
from magenta.offers import Arm, OfferDecision


def _report():
    return RiskUpliftReport(
        p_churn=0.8, band=Band.CRITICAL,
        drivers=[Driver(feature="OVERAGE_EVENTS", label="Overage",
                        shap_value=0.4, direction="UP")],
        tau_hat=0.22, segment=Segment.PERSUADABLE, engage=True, timing=Timing.ACT_NOW)


def _diag(conf=0.8):
    return Diagnosis(root_cause_tags=["BILL_SHOCK"], narrative="n",
                     eligible_offer_ids=[Arm.BILL_CREDIT.value, Arm.PLAN_DOWNSELL.value],
                     confidence=conf)


def test_should_deliberate_high_clv(customer):
    customer.clv_estimate = 5000.0
    assert should_deliberate(customer, _diag(conf=0.9), p90_clv=2000.0) is True


def test_should_deliberate_low_confidence(customer):
    customer.clv_estimate = 100.0
    assert should_deliberate(customer, _diag(conf=0.3), p90_clv=2000.0) is True


def test_should_not_deliberate_typical(customer):
    customer.clv_estimate = 100.0
    assert should_deliberate(customer, _diag(conf=0.9), p90_clv=2000.0) is False


class LargeChat:
    """Council returns arm names; critic returns PASS."""

    def __init__(self, critic="PASS"):
        self.roles = []
        self._critic = critic
        self._calls = 0

    def chat(self, role, messages, **kw):
        self.roles.append(role)
        self._calls += 1
        text = " ".join(str(m.get("content", "")) for m in messages).lower()
        if "critic" in text or "validate" in text:
            return self._critic
        if "network" in text:
            return "PLAN_DOWNSELL"
        return "BILL_CREDIT"


class Bandit:
    def posterior_mean(self, x, arm):
        return {Arm.BILL_CREDIT: 12.0, Arm.PLAN_DOWNSELL: 9.0}.get(arm, 1.0)

    def select(self, x, eligible):
        return eligible[0], 0.5


class Cat:
    def eligible(self, c):
        return [Arm.BILL_CREDIT, Arm.PLAN_DOWNSELL, Arm.DATA_BOOST]

    def cost(self, arm):
        return {Arm.BILL_CREDIT: 8.0, Arm.PLAN_DOWNSELL: 6.0}.get(arm, 5.0)

    def min_margin(self, arm):
        return 0.0


class Uplift:
    def tau(self, c):
        return 0.2


class Deps:
    def __init__(self, chat):
        self.chat = chat
        self.bandit = Bandit()
        self.catalog = Cat()
        self.uplift = Uplift()


def test_deliberate_three_to_four_calls_and_picks_best(customer, monkeypatch):
    import magenta.graph.system2 as s2
    monkeypatch.setattr(s2, "featurize", lambda c: [0.0])
    chat = LargeChat(critic="PASS")
    deps = Deps(chat)
    offer = deliberate(customer, _report(), _diag(), deps)
    assert isinstance(offer, OfferDecision)
    # lookahead: tau(0.2)*posterior => BILL_CREDIT (12) beats PLAN_DOWNSELL (9)
    assert offer.arm is Arm.BILL_CREDIT
    assert 3 <= chat._calls <= 4       # council(2) + critic(1) [+ optional]


def test_deliberate_critic_reject_falls_back_cheapest(customer, monkeypatch):
    import magenta.graph.system2 as s2
    monkeypatch.setattr(s2, "featurize", lambda c: [0.0])
    chat = LargeChat(critic="REJECT")
    deps = Deps(chat)
    offer = deliberate(customer, _report(), _diag(), deps)
    # critic rejects => cheapest eligible (PLAN_DOWNSELL cost 6 < BILL_CREDIT 8)
    assert offer.arm is Arm.PLAN_DOWNSELL
