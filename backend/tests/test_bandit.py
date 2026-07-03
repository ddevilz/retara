import sqlite3

import numpy as np

from magenta.brain.bandit import ThompsonBandit
from magenta.offers import Arm


def test_select_returns_eligible_arm_and_valid_propensity():
    b = ThompsonBandit(dim=4, arms=[Arm.NO_ACTION, Arm.BILL_CREDIT, Arm.DATA_BOOST])
    x = np.ones(4)
    arm, prop = b.select(x, eligible=[Arm.BILL_CREDIT, Arm.DATA_BOOST])
    assert arm in {Arm.BILL_CREDIT, Arm.DATA_BOOST}
    assert 0.0 < prop <= 1.0


def test_converges_on_two_arm_problem():
    # Arm A pays 1.0 for context x; arm B pays 0.0. Best arm share should dominate.
    rng = np.random.default_rng(0)
    arms = [Arm.BILL_CREDIT, Arm.DATA_BOOST]
    b = ThompsonBandit(dim=3, arms=arms)
    x = np.array([1.0, 0.0, 1.0])
    picks = []
    for _ in range(500):
        arm, _ = b.select(x, eligible=arms)
        reward = 1.0 if arm == Arm.BILL_CREDIT else 0.0
        reward += rng.normal(0, 0.05)
        b.update(x, arm, reward)
        picks.append(arm)
    last = picks[-100:]
    share = sum(a == Arm.BILL_CREDIT for a in last) / len(last)
    assert share > 0.7, f"best-arm share {share} too low"


def test_save_load_roundtrip():
    arms = [Arm.NO_ACTION, Arm.BILL_CREDIT]
    b = ThompsonBandit(dim=2, arms=arms)
    x = np.array([1.0, 1.0])
    b.update(x, Arm.BILL_CREDIT, 5.0)
    conn = sqlite3.connect(":memory:")
    b.save(conn)
    b2 = ThompsonBandit(dim=2, arms=arms)
    b2.load(conn)
    # posteriors identical -> same mean estimate.
    m1 = b._theta_mean(Arm.BILL_CREDIT)
    m2 = b2._theta_mean(Arm.BILL_CREDIT)
    assert np.allclose(m1, m2)
