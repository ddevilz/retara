from magenta.brain.risk import Band, RiskAssessment
from magenta.graph.ablation import RUNGS, make_policy
from magenta.graph.build import GraphDeps
from magenta.graph.policy import AgentPolicy
from magenta.offers import Arm


class Cust:
    def __init__(self, cid="C", p=0.7, overage=2, end=20):
        self.customer_id = cid
        self.overage_events = overage
        self.contract_end_days = end


class Risk:
    def __init__(self, p): self._p = p
    def score(self, c):
        return RiskAssessment(p_churn=self._p, band=Band.HIGH, drivers=[])


class Deps:  # minimal deps for rung factories that need risk
    def __init__(self, p=0.7):
        self.risk = Risk(p)


def test_rungs_order():
    assert RUNGS == ["noaction", "rules", "risk_rules", "agent_s1", "agent"]


def test_noaction_always_none():
    pol = make_policy("noaction", Deps())
    assert pol.decide(Cust()) is None


def test_rules_fires_on_overage():
    pol = make_policy("rules", Deps())
    o = pol.decide(Cust(overage=3, end=200))
    assert o.arm is Arm.BILL_CREDIT


def test_rules_none_when_calm():
    pol = make_policy("rules", Deps())
    assert pol.decide(Cust(overage=0, end=200)) is None


def test_risk_rules_gated_by_p_churn():
    hi = make_policy("risk_rules", Deps(p=0.8))
    lo = make_policy("risk_rules", Deps(p=0.2))
    assert hi.decide(Cust()).arm is Arm.BILL_CREDIT
    assert lo.decide(Cust()) is None


## ---- rung dispatch for the agent rungs (system2_enabled plumbing, 7.4/7.6) --

def _real_deps(customer, fakes, spy_chat, conn):
    return GraphDeps(
        risk=fakes["risk"], uplift=fakes["uplift"], bandit=fakes["bandit"],
        catalog=fakes["catalog"], oracle=fakes["oracle"], conn=conn,
        params=type("P", (), {"freq_cap_days": 14, "freq_cap_max": 1, "value_cap": 40.0})(),
        chat=spy_chat, load_customer=lambda cid: customer, checkpointer=None,
    )


def test_agent_s1_is_agent_policy_with_system2_disabled(customer, fakes, spy_chat, db_conn):
    pol = make_policy("agent_s1", _real_deps(customer, fakes, spy_chat, db_conn))
    assert isinstance(pol, AgentPolicy)
    assert pol.deps.system2_enabled is False


def test_agent_is_agent_policy_with_system2_enabled(customer, fakes, spy_chat, db_conn):
    pol = make_policy("agent", _real_deps(customer, fakes, spy_chat, db_conn))
    assert isinstance(pol, AgentPolicy)
    assert pol.deps.system2_enabled is True
