import time

import numpy as np
import pytest

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


def test_posterior_mean_matches_theta_mean_and_is_deterministic():
    # System-2's lookahead (magenta.graph.system2.deliberate) needs a
    # deterministic (non-sampled) posterior estimate to score candidate arms
    # reproducibly within one deliberation call — ThompsonBandit.select()'s
    # stochastic draw is unsuitable for that. posterior_mean must equal
    # x . theta_mean and must NOT vary across repeated calls.
    arms = [Arm.BILL_CREDIT, Arm.DATA_BOOST]
    b = ThompsonBandit(dim=3, arms=arms, seed=0)
    x = np.array([1.0, 2.0, 3.0])
    b.update(x, Arm.BILL_CREDIT, 5.0)

    expected = float(x @ b._theta_mean(Arm.BILL_CREDIT))
    first = b.posterior_mean(x, Arm.BILL_CREDIT)
    second = b.posterior_mean(x, Arm.BILL_CREDIT)

    assert first == pytest.approx(expected)
    assert first == second  # deterministic, unlike select()'s TS draw


def test_save_load_roundtrip(db_conn):
    from tests.db_fixtures import TENANT_A

    arms = [Arm.NO_ACTION, Arm.BILL_CREDIT]
    b = ThompsonBandit(dim=2, arms=arms)
    x = np.array([1.0, 1.0])
    b.update(x, Arm.BILL_CREDIT, 5.0)
    b.save(db_conn, TENANT_A)
    b2 = ThompsonBandit(dim=2, arms=arms)
    b2.load(db_conn, TENANT_A)
    # posteriors identical -> same mean estimate.
    m1 = b._theta_mean(Arm.BILL_CREDIT)
    m2 = b2._theta_mean(Arm.BILL_CREDIT)
    assert np.allclose(m1, m2)


def test_posterior_roundtrip_preserves_float64(db_conn):
    """The blob carries no dtype metadata — reshape(dim, dim) on the wrong dtype
    yields garbage rather than an error. This asserts the bytes survive exactly."""
    from tests.db_fixtures import TENANT_A

    b = ThompsonBandit(dim=4, arms=list(Arm), seed=1)
    # NOTE: the brief's draft test called update(arm, x, reward=...), but the
    # real ThompsonBandit.update signature (see above, and every other test in
    # this file) is update(x, arm, reward) — x first. Fixed here to match the
    # real code rather than changing update() to match the brief.
    b.update(np.array([1.0, 2.0, 3.0, 4.0]), Arm.BILL_CREDIT, reward=1.0)
    b.save(db_conn, TENANT_A)

    restored = ThompsonBandit(dim=4, arms=list(Arm), seed=1)
    restored.load(db_conn, TENANT_A)

    np.testing.assert_array_equal(restored._A[Arm.BILL_CREDIT], b._A[Arm.BILL_CREDIT])
    np.testing.assert_array_equal(restored._b[Arm.BILL_CREDIT], b._b[Arm.BILL_CREDIT])
    assert restored._A[Arm.BILL_CREDIT].dtype == np.float64


def test_posteriors_are_tenant_isolated(db_conn):
    from tests.db_fixtures import TENANT_A, TENANT_B

    a = ThompsonBandit(dim=4, arms=list(Arm), seed=1)
    a.update(np.array([1.0, 2.0, 3.0, 4.0]), Arm.BILL_CREDIT, reward=1.0)
    a.save(db_conn, TENANT_A)

    b = ThompsonBandit(dim=4, arms=list(Arm), seed=1)
    b.load(db_conn, TENANT_B)  # no rows for B — must stay at the prior
    assert b._n[Arm.BILL_CREDIT] == 0


def test_load_does_not_leave_connection_in_open_transaction(db_conn):
    """A read-only load() that never commits leaves its connection idle in
    transaction. On a long-lived connection (api.deps.get_graph_deps()'s
    @lru_cache singleton) that ACCESS-SHARE-locks BANDIT_POSTERIOR forever,
    blocking any later TRUNCATE — the bug that made the full suite hang."""
    from tests.db_fixtures import TENANT_A

    b = ThompsonBandit(dim=4, arms=list(Arm), seed=1)
    b.load(db_conn, TENANT_A)
    assert not db_conn.in_transaction()


def test_select_many_calls_stays_fast():
    """select() used to draw its 101 samples per arm one at a time via
    SVD-based multivariate_normal — ~700k SVDs for a realistic cohort run,
    several minutes of wall clock that read as a hang under any reasonable
    test timeout. Batching into one Cholesky-based draw per arm made 300
    select()+update() calls take ~0.5s; this asserts a generous 5s bound for
    200 calls so a regression back to per-draw SVD fails loudly instead of
    reading as flakiness."""
    arms = [Arm.BILL_CREDIT, Arm.DATA_BOOST, Arm.PLAN_DOWNSELL]
    b = ThompsonBandit(dim=6, arms=arms, seed=2)
    x = np.random.default_rng(0).random(6)

    t0 = time.perf_counter()
    for _ in range(200):
        arm, _ = b.select(x, eligible=arms)
        b.update(x, arm, reward=1.0)
    assert time.perf_counter() - t0 < 5.0
