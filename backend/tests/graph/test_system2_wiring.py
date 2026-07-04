from unittest.mock import patch

from magenta.graph import nodes as N
from magenta.offers import Arm, OfferDecision


class _Deps:  # minimal decide-node deps
    def __init__(self, s2: bool):
        self.system2_enabled = s2
        self.catalog = _Catalog()
        self.bandit = _Bandit()
        self.load_customer = lambda cid: _Cust()
        self.params = type("P", (), {"p90_clv": 100.0})()  # low -> always triggers


class _Cust:
    customer_id = "C1"
    clv_estimate = 500.0  # >= p90_clv -> should_deliberate True


class _Catalog:
    def eligible(self, c):
        return [Arm.NO_ACTION, Arm.BILL_CREDIT]

    def cost(self, arm):
        return 8.0


class _Bandit:
    def select(self, x, eligible):
        return Arm.NO_ACTION, 0.9


def _state():
    from magenta.graph.state import Diagnosis
    return {
        "customer_id": "C1",
        "risk": None,
        "diagnosis": Diagnosis(root_cause_tags=["BILL_SHOCK"], narrative="n",
                               eligible_offer_ids=[Arm.NO_ACTION.value,
                                                   Arm.BILL_CREDIT.value],
                               confidence=0.9),
    }


def test_decide_uses_system2_when_enabled():
    s2_offer = OfferDecision(arm=Arm.BILL_CREDIT, cost=8.0,
                             rationale="system2: n", propensity=1.0)
    with patch("magenta.graph.system2.deliberate", return_value=s2_offer) as d:
        out = N.decide(_state(), deps=_Deps(s2=True))
    d.assert_called_once()
    assert out["offer"].arm is Arm.BILL_CREDIT
    assert out["audit_log"][0]["NODE"] == "DECIDE_S2"


def test_decide_skips_system2_when_disabled(monkeypatch):
    # NOTE (brief bug fixed on sight): _Cust only stubs customer_id/clv_estimate,
    # but the bandit-path fallback calls featurize(customer), which needs the
    # full Customer numeric surface (tenure_months, monthly_charge, ...). The
    # brief's snippet never exercised this path against the real featurize —
    # monkeypatch it like the existing decide()-path tests in test_nodes.py do.
    monkeypatch.setattr(N, "featurize", lambda c: [0.0])
    with patch("magenta.graph.system2.deliberate") as d:
        out = N.decide(_state(), deps=_Deps(s2=False))
    d.assert_not_called()
    assert out["audit_log"][0]["NODE"] == "DECIDE"
