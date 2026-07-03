from types import SimpleNamespace

import numpy as np
import pytest

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
    with pytest.raises(Exception):
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
