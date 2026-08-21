from unittest.mock import patch

from magenta.evalx.golden import (
    GOLDEN_SCENARIOS,
    GoldenResult,
    run_golden,
)


def test_at_least_ten_scenarios_with_required_names():
    assert len(GOLDEN_SCENARIOS) >= 10
    names = {s.name for s in GOLDEN_SCENARIOS}
    for required in ["sleeping_dog_no_contact", "bill_shock_gets_ack_or_credit",
                     "network_complainer_not_offered_discount", "holdout_never_fulfilled"]:
        assert required in names


def test_run_golden_passes_on_good_policy():
    # stub the per-scenario evaluator so the graph isn't actually run here
    with patch("magenta.evalx.golden._evaluate",
               side_effect=lambda s: GoldenResult(name=s.name, passed=True, detail="ok")):
        results = run_golden()
    assert all(r.passed for r in results)
    assert len(results) == len(GOLDEN_SCENARIOS)


def test_run_golden_catches_broken_policy():
    # deliberately break the network-complainer scenario → regression demo
    def broken(s):
        bad = s.name == "network_complainer_not_offered_discount"
        return GoldenResult(name=s.name, passed=not bad,
                            detail="offered discount to network complainer" if bad else "ok")
    with patch("magenta.evalx.golden._evaluate", side_effect=broken):
        results = run_golden()
    failed = [r for r in results if not r.passed]
    assert len(failed) == 1
    assert failed[0].name == "network_complainer_not_offered_discount"


def test_run_golden_all_scenarios_pass_on_real_graph():
    # NO patching here: exercises the real _evaluate -> run_scenario -> compiled
    # graph path (in-memory sqlite conn + rule-based deterministic chat stub,
    # per the brief's "deterministic diagnosis per scenario"). This is the
    # actual regression pin — the three tests above only prove the harness
    # *structure* is patchable; this one proves the *behavior* is correct today
    # so a future change that breaks a disposition fails loudly here.
    results = run_golden()
    failed = [(r.name, r.detail) for r in results if not r.passed]
    assert not failed, f"golden scenario regressions: {failed}"
    assert len(results) == len(GOLDEN_SCENARIOS)
