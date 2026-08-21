from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from magenta.config import configs_dir
from magenta.sim.oracle import Outcome, ResponseOracle, SimParams
from magenta.sim.population import Segment, generate_population


def _params() -> SimParams:
    return SimParams.load(configs_dir() / "sim_params.yaml")


def _stub_offer(arm="BILL_CREDIT", cost=8.0, fits=("BILL_SHOCK",)):
    # duck-typed stand-in for OfferDecision (Lab 2). Oracle only reads these attrs.
    return SimpleNamespace(arm=arm, cost=cost, fits_causes=list(fits))


def test_simparams_is_frozen():
    p = _params()
    with pytest.raises(ValidationError):
        p.churn_A0 = 0.0  # frozen model -> mutation raises


def test_outcome_shape():
    pop, hid = generate_population(20, seed=1)
    oracle = ResponseOracle(hid, _params(), seed=42)
    out = oracle.outcome(pop[0], None)
    assert isinstance(out, Outcome)
    assert isinstance(out.accepted, bool)
    assert isinstance(out.churned, bool)


def test_crn_paired_reproducible():
    pop, hid = generate_population(20, seed=1)
    o1 = ResponseOracle(hid, _params(), seed=7)
    o2 = ResponseOracle(hid, _params(), seed=7)
    for c in pop:
        a = o1.outcome(c, None)
        b = o2.outcome(c, None)
        assert (a.accepted, a.churned) == (b.accepted, b.churned)


def _churn_prob(oracle, customer, offer):
    # helper on the oracle for probability inspection (deterministic, no draw)
    return oracle.churn_prob(customer, offer)


def test_sleeping_dog_contact_increases_churn_prob():
    pop, hid = generate_population(4000, seed=2)
    oracle = ResponseOracle(hid, _params(), seed=3)
    dogs = [c for c in pop
            if hid[c.customer_id].persuadable_segment == Segment.SLEEPING_DOG]
    assert dogs, "expected sleeping dogs in population"
    offer = _stub_offer()
    worse = 0
    for c in dogs[:300]:
        p_no = _churn_prob(oracle, c, None)
        p_yes = _churn_prob(oracle, c, offer)
        if p_yes > p_no:
            worse += 1
    # contacting a sleeping dog should raise churn prob for the large majority
    assert worse / len(dogs[:300]) > 0.9


def test_persuadable_good_fit_offer_decreases_churn_prob():
    pop, hid = generate_population(4000, seed=4)
    oracle = ResponseOracle(hid, _params(), seed=5)
    pers = [c for c in pop
            if hid[c.customer_id].persuadable_segment == Segment.PERSUADABLE]
    assert pers
    offer = _stub_offer(arm="BILL_CREDIT", cost=8.0, fits=("BILL_SHOCK",))
    better = 0
    for c in pers[:300]:
        p_no = _churn_prob(oracle, c, None)
        p_yes = _churn_prob(oracle, c, offer)
        if p_yes < p_no:
            better += 1
    assert better / len(pers[:300]) > 0.9


# ---- Finding 1: sleeping-dog penalty must fire on CONTACT, not acceptance ----

def test_sleeping_dog_harmed_even_when_declining(monkeypatch):
    """Governing spec: "sleeping-dogs get a positive contact_penalty ->
    contacting them increases churn." The harm is the ANNOYANCE of being
    contacted, not a consequence of saying yes. Force every sleeping dog to
    DECLINE (monkeypatch accept_prob -> 0.0, so `accept_rng.random() < 0.0`
    never fires) and confirm the simulated churn RATE for contacted-but-
    declining sleeping dogs is still measurably higher than never-contacted.

    Needs a large sleeping-dog sample (>=2000) because we're comparing
    stochastic Bernoulli outcome() rates, not the underlying deterministic
    probabilities -- a small sample would be too noisy to reliably show the
    direction. Population size (30000) picked so the ~8% SLEEPING_DOG mix
    yields >=2000 dogs at this fixed seed.
    """
    pop, hid = generate_population(30000, seed=21)
    oracle = ResponseOracle(hid, _params(), seed=23)
    dogs = [c for c in pop
            if hid[c.customer_id].persuadable_segment == Segment.SLEEPING_DOG]
    assert len(dogs) >= 2000, f"need >=2000 sleeping dogs for a stable rate estimate, got {len(dogs)}"

    monkeypatch.setattr(oracle, "accept_prob", lambda customer, offer: 0.0)
    offer = _stub_offer()

    churn_never = 0
    churn_declined = 0
    for c in dogs:
        out_never = oracle.outcome(c, None)
        out_declined = oracle.outcome(c, offer)
        assert out_declined.accepted is False, "test setup: acceptance must be forced off"
        churn_never += out_never.churned
        churn_declined += out_declined.churned

    rate_never = churn_never / len(dogs)
    rate_declined = churn_declined / len(dogs)
    assert rate_declined > rate_never, (
        f"contacted-but-declined churn rate {rate_declined:.4f} should exceed "
        f"never-contacted {rate_never:.4f} -- contact itself, not acceptance, "
        f"must harm sleeping dogs"
    )


# ---- Finding 2: CRN cross-arm pairing regression tests ----

def test_crn_cross_arm_pairing():
    """White-box regression for the CRN pairing property: the "churn" stream
    draw for a given customer_id+seed must be IDENTICAL whether offer=None
    or offer=<anything> -- that's what makes treatment/holdout comparable
    (same random draw, only the probability it's compared against differs).

    This guards against a silent regression back to a single shared rng per
    customer (the plan's original reference code drew accept, then churn,
    sequentially off ONE generator) which would shift the churn draw's
    position whenever an offer is present and consumed an accept draw first
    -- exactly the bug the current per-stream `_rng(cid, stream)` design
    avoids, but which had no test pinning it down.
    """
    pop, hid = generate_population(200, seed=31)
    oracle = ResponseOracle(hid, _params(), seed=37)
    offer = _stub_offer()

    for c in pop:
        cid = c.customer_id
        # the base-churn draw, computed independently of any arm/outcome() call
        u = oracle._rng(cid, "churn").random()

        out_no = oracle.outcome(c, None)
        p_no = oracle.churn_prob(c, None)
        assert out_no.churned == (u < p_no)

        out_yes = oracle.outcome(c, offer)
        p_yes = oracle.churn_prob(c, offer, accepted=out_yes.accepted)
        assert out_yes.churned == (u < p_yes)


def test_crn_cross_arm_pairing_black_box():
    """Black-box companion to test_crn_cross_arm_pairing: for non-SLEEPING_DOG
    customers, offer=None vs a ZERO-EFFECT dummy offer (no cost, no
    fits_causes) should produce the SAME churned boolean almost always.

    Tolerance, not exact equality: the dummy offer still carries a nonzero
    ACCEPT probability (accept logit's G0 intercept and tenure-trust term
    don't depend on offer value/fit at all -- e.g. sigmoid(G0) ~= 0.40 accept
    chance even for a customer with zero tenure and zero price sensitivity),
    and once accepted, the "some help even without exact fit" 0.35 fallback
    still nudges churn_prob down via segment_responsiveness for
    PERSUADABLE/SURE_THING/LOST_CAUSE -- enough, for any customer whose fixed
    "churn" draw sits between the offer=None probability and the
    slightly-lower dummy-accepted probability, to flip the boolean. That is
    a real, expected effect of the offer/accept channel (not a CRN bug), and
    empirically flips ~2.2% of non-dog customers at this population/seed
    (measured match rate ~0.978) -- so the threshold is set at >=0.97 (a
    safety margin below the measured baseline, not a freshly-invented
    number) rather than the tighter ~99% initially assumed. SLEEPING_DOG is
    excluded here on purpose: this test is about the *benefit* path only
    (segment_responsiveness[SLEEPING_DOG] is frozen at 0.0 so it never has a
    benefit-driven mismatch), the *contact-penalty* path is covered
    separately above.
    """
    pop, hid = generate_population(2000, seed=41)
    oracle = ResponseOracle(hid, _params(), seed=43)
    non_dogs = [c for c in pop
                if hid[c.customer_id].persuadable_segment != Segment.SLEEPING_DOG]
    assert len(non_dogs) > 1000
    dummy = _stub_offer(arm="NO_ACTION", cost=0.0, fits=())

    matches = 0
    for c in non_dogs:
        out_no = oracle.outcome(c, None)
        out_dummy = oracle.outcome(c, dummy)
        if out_no.churned == out_dummy.churned:
            matches += 1
    match_rate = matches / len(non_dogs)
    assert match_rate >= 0.97, f"expected >=97% cross-arm match for non-dogs, got {match_rate:.4f}"
