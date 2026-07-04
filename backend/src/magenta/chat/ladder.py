from __future__ import annotations

from magenta.offers import Offer, OfferCatalog, OfferDecision
from magenta.graph.state import Diagnosis
from magenta.chat.state import DialogueState


def _rationale(offer: Offer, rung: int) -> str:
    return f"Ladder rung {rung}: {offer.arm.value} (cost EUR {offer.cost:.0f})."


class OfferLadder:
    def __init__(self, catalog: OfferCatalog, diagnosis: Diagnosis, authority_cap_eur: float):
        self.authority_cap_eur = authority_cap_eur
        self._diagnosis = diagnosis
        tags = set(diagnosis.root_cause_tags)
        offers_to_filter = catalog.offers.values() if isinstance(catalog.offers, dict) else catalog.offers
        fitting = [o for o in offers_to_filter if tags & set(o.fits_causes)]
        self._rungs: list[Offer] = sorted(fitting, key=lambda o: o.cost)
        self._position = 0  # 0 = un-opened; else 1..len(rungs)

    @property
    def position(self) -> int:
        return self._position

    def _decision(self, offer: Offer) -> OfferDecision:
        return OfferDecision(
            arm=offer.arm,
            cost=offer.cost,
            rationale=_rationale(offer, self._position),
            propensity=1.0,
        )

    def open(self) -> OfferDecision:
        if not self._rungs:
            raise ValueError("no fitting offers on ladder for diagnosis")
        self._position = 1
        return self._decision(self._rungs[0])

    def concede(self, state: DialogueState) -> OfferDecision | None:
        next_idx = self._position  # 0-based index of the next rung
        if next_idx >= len(self._rungs):
            return None
        nxt = self._rungs[next_idx]
        if nxt.cost > self.authority_cap_eur:
            return None
        self._position += 1
        return self._decision(nxt)
