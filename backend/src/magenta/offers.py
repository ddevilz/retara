"""Offer catalog — 8 arms with costs, margins, eligibility, cause-fit.

OfferDecision is the agent's output at the Act node; the oracle reads arm/cost/fits_causes.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel

from magenta.config import load_yaml
from magenta.sim.population import Customer


class Arm(str, Enum):
    NO_ACTION = "NO_ACTION"
    ACKNOWLEDGE_AND_FIX = "ACKNOWLEDGE_AND_FIX"
    BILL_CREDIT = "BILL_CREDIT"
    PLAN_DOWNSELL = "PLAN_DOWNSELL"
    DATA_BOOST = "DATA_BOOST"
    DEVICE_UPGRADE = "DEVICE_UPGRADE"
    NETWORK_PRIORITY_FIX = "NETWORK_PRIORITY_FIX"
    BUNDLE_ADDON = "BUNDLE_ADDON"


class Offer(BaseModel):
    arm: Arm
    cost: float
    min_margin: float
    eligibility_note: str
    fits_causes: list[str]


class OfferDecision(BaseModel):
    arm: Arm
    cost: float
    rationale: str = ""
    propensity: float = 1.0


class OfferCatalog(BaseModel):
    offers: dict[Arm, Offer]

    @classmethod
    def load(cls, path: str | Path) -> "OfferCatalog":
        raw = load_yaml(Path(path))
        offers: dict[Arm, Offer] = {}
        for name, spec in raw["arms"].items():
            arm = Arm(name)
            offers[arm] = Offer(
                arm=arm,
                cost=float(spec["cost"]),
                min_margin=float(spec["min_margin"]),
                eligibility_note=str(spec["eligibility_note"]),
                fits_causes=list(spec.get("fits_causes", [])),
            )
        missing = set(Arm) - set(offers)
        if missing:
            raise ValueError(f"offers.yaml missing arms: {sorted(a.value for a in missing)}")
        return cls(offers=offers)

    def get(self, arm: Arm) -> Offer:
        return self.offers[arm]

    def cost(self, arm: Arm) -> float:
        return self.offers[arm].cost

    def eligible(self, customer: Customer) -> list[Arm]:
        """Arms this customer may receive. Deterministic; margin-safe; NO_ACTION always in."""
        out: list[Arm] = [Arm.NO_ACTION]
        for arm in Arm:
            if arm == Arm.NO_ACTION:
                continue
            off = self.offers[arm]
            # margin floor: offer must leave at least min_margin of monthly margin
            if customer.gross_margin_monthly - off.cost < off.min_margin:
                continue
            # arm-specific eligibility
            if arm == Arm.DEVICE_UPGRADE and customer.contract_end_days > 90:
                continue
            if arm == Arm.DATA_BOOST and customer.data_gb_used_p50 < customer.data_allowance_gb * 0.6:
                continue
            if arm == Arm.PLAN_DOWNSELL and customer.plan == "BASIC":
                continue
            out.append(arm)
        return out
