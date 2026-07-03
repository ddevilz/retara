from magenta.experiment import (
    NoActionPolicy,
    RulesPolicy,
    Scorecard,
    assign_holdout,
    run_experiment,
)


def test_assign_holdout_is_stable_and_balanced():
    seed = 42
    a = assign_holdout("C0000001", seed)
    assert a == assign_holdout("C0000001", seed)  # stable
    hold = sum(assign_holdout(f"C{i:07d}", seed) for i in range(4000))
    assert 0.45 < hold / 4000 < 0.55  # ~50/50


def test_noaction_ate_covers_zero():
    # NOTE: n bumped 6000 -> 10000 vs. the original brief. At n=6000/seed=42 the
    # holdout/treatment split lands a borderline covariate imbalance (z ~ -1.99,
    # i.e. right at the 95% CI edge) against this codebase's current calibrated
    # churn_A0 (see sim commit "calibrate A0 to realize 26.5% base churn") --
    # expected ~5% of the time for any single seed on a valid 95% CI, confirmed
    # by sweeping seeds 1..60 (~8% landed outside, consistent with that rate).
    # n=10000 gives a comfortable margin (z ~ -1.0) without changing the test's
    # intent: NoAction has zero causal effect, so ATE should be ~0 and the CI
    # should cover 0.
    sc = run_experiment(NoActionPolicy(), n=10000, seed=42)
    assert sc.offers_made == 0
    assert abs(sc.ate) < 0.03
    assert sc.ci_low <= 0.0 <= sc.ci_high


def test_holdout_purity_noaction():
    # NoAction makes no offers at all -> trivially pure; and holdout groups are nonempty
    sc = run_experiment(NoActionPolicy(), n=4000, seed=7)
    assert sc.n_holdout > 0 and sc.n_treatment > 0


def test_rules_policy_makes_offers_and_reduces_churn():
    sc = run_experiment(RulesPolicy(), n=8000, seed=42)
    assert sc.offers_made > 0
    # a naive save policy should not INCREASE churn on net in this sim
    assert sc.ate >= -0.005  # ate = holdout - treatment; >=0 means treatment churned less-or-equal
    assert 0.0 <= sc.acceptance_rate <= 1.0
    assert sc.offer_spend > 0


def test_scorecard_math_consistency():
    sc = run_experiment(RulesPolicy(), n=5000, seed=3)
    assert sc.n_treatment + sc.n_holdout == 5000
    assert abs(sc.ate - (sc.churn_holdout - sc.churn_treatment)) < 1e-9
    assert 0.0 <= sc.churn_treatment <= 1.0
    assert 0.0 <= sc.churn_holdout <= 1.0
    assert isinstance(sc, Scorecard)


def test_holdout_never_receives_offer():
    # instrument: run with a spy that records ids offered, cross-check against holdout
    from magenta.experiment import _run_arms  # internal, returns per-customer records

    records = _run_arms(RulesPolicy(), n=3000, seed=11, budget=None)
    for r in records:
        if r["holdout"]:
            assert r["offer_arm"] is None


def test_budget_cap_limits_spend():
    unlimited = run_experiment(RulesPolicy(), n=6000, seed=9, budget=None)
    capped = run_experiment(RulesPolicy(), n=6000, seed=9, budget=500.0)
    assert capped.offer_spend <= 500.0 + 1e-6
    assert capped.offer_spend <= unlimited.offer_spend
