"""L3 response oracle — stochastic churn (discrete-time hazard) + accept (logistic),
with the SLEEPING_DOG contact penalty. The agent NEVER sees this; it returns only
the Outcome. CRN: a per-(customer_id, seed) rng makes treatment/holdout paired.

Offer typing is duck-typed to avoid a hard import cycle with magenta.offers (Lab 2):
the oracle reads only offer.arm, offer.cost, offer.fits_causes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict

from magenta.config import load_yaml
from magenta.sim.events import generate_events
from magenta.sim.population import Customer, HiddenStore, Segment

if TYPE_CHECKING:  # pragma: no cover
    pass


# cause-fit map: which event causes each arm addresses (mirrors offers.yaml fits_causes)
_ARM_FITS: dict[str, set[str]] = {
    "NO_ACTION": set(),
    "ACKNOWLEDGE_AND_FIX": {"SERVICE_COMPLAINT", "DROPPED_CALLS"},
    "BILL_CREDIT": {"BILL_SHOCK", "OVERAGE"},
    "PLAN_DOWNSELL": {"BILL_SHOCK", "OVERAGE"},
    "DATA_BOOST": {"OVERAGE"},
    "DEVICE_UPGRADE": {"CONTRACT_EXPIRY"},
    "NETWORK_PRIORITY_FIX": {"DROPPED_CALLS"},
    "BUNDLE_ADDON": {"COMPETITOR_OFFER", "CONTRACT_EXPIRY"},
}


class Outcome(BaseModel):
    accepted: bool
    churned: bool


class SimParams(BaseModel):
    """FROZEN behavioural coefficients loaded from sim_params.yaml."""

    model_config = ConfigDict(frozen=True)

    churn_A0: float
    churn_A1: float
    churn_A2: float
    churn_A3: float
    churn_BETA_OFFER: float

    accept_G0: float
    accept_G1: float
    accept_G2: float
    accept_G3: float
    accept_G4: float

    segment_responsiveness: dict[str, float]
    sleeping_dog_contact_penalty: float
    trust_tenure_halflife_months: float
    fatigue_per_recent_contact: float

    @classmethod
    def load(cls, path: str | Path) -> SimParams:
        raw = load_yaml(Path(path))
        churn = raw["churn"]
        accept = raw["accept"]
        return cls(
            churn_A0=churn["A0"],
            churn_A1=churn["A1"],
            churn_A2=churn["A2"],
            churn_A3=churn["A3"],
            churn_BETA_OFFER=churn["BETA_OFFER"],
            accept_G0=accept["G0"],
            accept_G1=accept["G1"],
            accept_G2=accept["G2"],
            accept_G3=accept["G3"],
            accept_G4=accept["G4"],
            segment_responsiveness=raw["segment_responsiveness"],
            sleeping_dog_contact_penalty=raw["sleeping_dog_contact_penalty"],
            trust_tenure_halflife_months=raw["trust_tenure_halflife_months"],
            fatigue_per_recent_contact=raw["fatigue_per_recent_contact"],
        )


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


class ResponseOracle:
    def __init__(self, hidden: HiddenStore, params: SimParams, seed: int):
        self.hidden = hidden
        self.params = params
        self.seed = seed

    # ---- CRN: deterministic rng per (customer_id, seed, stream) ----
    # Each logical draw (events, offer-fit, accept, base-churn) gets its OWN
    # stream (distinct suffix) so that consuming one stream never shifts the
    # position of another. This is what makes treatment/holdout paired: the
    # base churn draw for a given customer+seed is byte-for-byte identical
    # whether or not an offer is passed, because it never shares a stream
    # with the offer-dependent accept draw.
    def _rng(self, customer_id: str, stream: str = "") -> np.random.Generator:
        key = f"{customer_id}:{self.seed}:{stream}" if stream else f"{customer_id}:{self.seed}"
        h = hashlib.sha256(key.encode()).digest()
        sub = int.from_bytes(h[:8], "big")
        return np.random.default_rng(sub)

    def _offer_effect(self, customer: Customer, offer, include_benefit: bool = True) -> float:
        """Signed contribution of `offer` to the churn logit, split into two
        INDEPENDENTLY-gated components (Finding 1 fix):

        - contact_penalty (SLEEPING_DOG only): fires whenever this is called
          at all, i.e. whenever `offer is not None` in churn_prob. Contact
          itself — not acceptance — is what harms a sleeping dog (the
          annoyance / wake-up effect from the governing spec: "sleeping-dogs
          get a positive contact_penalty -> contacting them increases
          churn"). Always included regardless of `include_benefit`.
        - benefit (segment_responsiveness * cause-fit): only included when
          `include_benefit` is True (i.e. the offer was actually ACCEPTED —
          an unaccepted offer can't help). Additive with contact_penalty,
          not either/or: an accepted offer to a sleeping dog still pays the
          full contact penalty (segment_responsiveness[SLEEPING_DOG] == 0.0
          in the frozen YAML makes its own benefit zero anyway, but the two
          terms are computed independently on purpose).

        Negative net value => reduces churn (good save). Positive => increases
        churn (contact penalty dominates, e.g. a sleeping dog).
        """
        hs = self.hidden[customer.customer_id]
        seg = hs.persuadable_segment

        contact_penalty = (
            self.params.sleeping_dog_contact_penalty if seg == Segment.SLEEPING_DOG else 0.0
        )

        benefit = 0.0
        if include_benefit:
            resp = self.params.segment_responsiveness.get(seg.value, 0.0)
            arm = getattr(offer, "arm", None)
            arm_name = str(getattr(arm, "value", arm))
            fits = set(getattr(offer, "fits_causes", [])) or _ARM_FITS.get(arm_name, set())

            # cause-fit: does the offer address this customer's live event causes?
            # own stream ("effect") — independent of the base-churn "events" stream
            # and the "accept"/"churn" streams, so it never perturbs any of them.
            rng = self._rng(customer.customer_id, "effect")
            events = generate_events(customer, hs, rng)
            live_causes = {e.kind.value for e in events}
            fit = 1.0 if (fits & live_causes) else 0.35  # some help even without exact fit

            # positive magnitude of help, then made negative (reduces churn)
            magnitude = resp * fit
            benefit = -magnitude

        return contact_penalty + benefit

    def churn_prob(self, customer: Customer, offer, accepted: bool | None = None) -> float:
        """Deterministic churn probability (no random draw) — for tests/analysis.

        `accepted` gates only the BENEFIT half of the offer effect:
          - None (default) or True: benefit included — the historical "what
            would this offer do" analysis view for callers that don't model
            acceptance explicitly (e.g. the sanity tests below).
          - False: benefit excluded (a declined offer can't help), but the
            SLEEPING_DOG contact penalty still applies — contact, not
            acceptance, triggers it. This is what `outcome()` passes.
        """
        p = self.params
        hs = self.hidden[customer.customer_id]
        rng = self._rng(customer.customer_id, "events")
        events = generate_events(customer, hs, rng)
        event_sum = sum(e.hazard_multiplier for e in events)

        logit = (
            p.churn_A0
            + p.churn_A1 * hs.theta_churn_base
            + p.churn_A2 * event_sum
            + p.churn_A3 * hs.competitor_pull
        )
        if offer is not None:
            offer_effect = self._offer_effect(
                customer, offer, include_benefit=accepted is not False
            )
            logit += p.churn_BETA_OFFER * (-offer_effect)
            # NOTE: offer_effect is negative-for-help; BETA_OFFER is negative;
            # multiply by -effect so a good (negative) effect * negative BETA
            # nets to reduced churn. Sleeping-dog contact penalty (positive
            # effect) raises churn regardless of `accepted`.
        return float(_sigmoid(logit))

    def accept_prob(self, customer: Customer, offer) -> float:
        if offer is None:
            return 0.0
        p = self.params
        hs = self.hidden[customer.customer_id]
        offer_value_norm = float(getattr(offer, "cost", 0.0)) / 50.0
        arm = getattr(offer, "arm", None)
        arm_name = str(getattr(arm, "value", arm))
        fits = set(getattr(offer, "fits_causes", [])) or _ARM_FITS.get(arm_name, set())
        rng = self._rng(customer.customer_id, "events")
        live = {e.kind.value for e in generate_events(customer, hs, rng)}
        offer_fit = 1.0 if (fits & live) else 0.0
        trust = customer.tenure_months / p.trust_tenure_halflife_months
        fatigue = 0.0  # single-period MVP: no prior contacts
        logit = (
            p.accept_G0
            + p.accept_G1 * hs.theta_price_sens * offer_value_norm
            + p.accept_G2 * offer_fit
            + p.accept_G3 * trust
            - p.accept_G4 * fatigue
        )
        return float(_sigmoid(logit))

    def outcome(self, customer: Customer, offer) -> Outcome:
        """Draw accept + churn using per-customer CRN streams.

        CRN pairing: the base churn draw always comes from the customer's
        "churn" stream, which is never touched by the accept draw (its own
        "accept" stream). So calling outcome(c, None) and outcome(c, offer)
        for the same customer_id+seed consumes the identical base-churn
        random number in both cases — only the churn *probability* fed into
        that draw differs (via offer_effect), never the draw itself. That is
        the variance-reduction property CRN depends on; sharing one rng
        sequentially between the accept draw and the churn draw would shift
        the churn draw's position whenever an offer is present and break it.

        Finding 1 fix: we pass `offer` itself through to churn_prob (never
        nulled) together with the realized `accepted` flag, instead of the
        old `offer if accepted else None`. Nulling the offer entirely on
        decline silently dropped the SLEEPING_DOG contact penalty for anyone
        who said no — contact (offer is not None), not acceptance, is what
        must trigger that penalty. `churn_prob` uses `accepted` only to gate
        the separate BENEFIT term (an unaccepted offer still can't help).
        """
        accepted = False
        if offer is not None:
            accept_rng = self._rng(customer.customer_id, "accept")
            accepted = bool(accept_rng.random() < self.accept_prob(customer, offer))
        churn_rng = self._rng(customer.customer_id, "churn")
        churned = bool(churn_rng.random() < self.churn_prob(customer, offer, accepted=accepted))
        return Outcome(accepted=accepted, churned=churned)
