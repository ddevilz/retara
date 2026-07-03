"""BrainPolicy: engage-gate -> bandit arm selection -> greedy budget cap.

Implements the experiment ``Policy`` protocol. Uses observable features only.
Budget is greedy: offers are made until the remaining budget cannot cover the
cost; cohort prioritisation by tau*clv is the caller's ordering, but the policy
self-guards so it never overspends regardless of order.
"""
from __future__ import annotations

from math import inf

from magenta.brain.bandit import ThompsonBandit
from magenta.brain.features import featurize
from magenta.brain.risk import RiskModel
from magenta.brain.uplift import UpliftModel, classify_segment
from magenta.offers import Arm, OfferCatalog, OfferDecision
from magenta.sim.population import Customer, Segment

_RISK_FLOOR = 0.25


class BrainPolicy:
    def __init__(
        self,
        risk: RiskModel,
        uplift: UpliftModel,
        bandit: ThompsonBandit,
        catalog: OfferCatalog,
        budget: float | None = None,
    ) -> None:
        self.risk = risk
        self.uplift = uplift
        self.bandit = bandit
        self.catalog = catalog
        self.budget = budget
        self._remaining: float = inf if budget is None else float(budget)

    def reset_budget(self) -> None:
        self._remaining = inf if self.budget is None else float(self.budget)

    def priority(self, c: Customer) -> float:
        """tau * CLV — the greedy spend ordering key (caller may sort by this)."""
        return self.uplift.tau(c) * float(c.clv_estimate)

    def _cost(self, arm: Arm) -> float:
        if hasattr(self.catalog, "cost"):
            return float(self.catalog.cost(arm))
        return 0.0

    def decide(self, c: Customer) -> OfferDecision | None:
        assessment = self.risk.score(c)
        tau = self.uplift.tau(c)
        segment = classify_segment(assessment.p_churn, tau)

        # Engage-gate: only persuadable + above risk floor get an offer.
        if assessment.p_churn < _RISK_FLOOR or segment is not Segment.PERSUADABLE:
            return None

        eligible = [a for a in self.catalog.eligible(c) if a != Arm.NO_ACTION]
        # Respect budget: drop arms we cannot afford.
        affordable = [a for a in eligible if self._cost(a) <= self._remaining]
        if not affordable:
            return None

        x = featurize(c)
        arm, propensity = self.bandit.select(x, eligible=affordable)
        cost = self._cost(arm)
        self._remaining -= cost
        return OfferDecision(
            arm=arm,
            cost=cost,
            rationale=f"persuadable p_churn={assessment.p_churn:.2f} tau={tau:.3f}",
            propensity=propensity,
        )
