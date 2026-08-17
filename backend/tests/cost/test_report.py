def test_cohort_uses_cache_and_reports(monkeypatch, small_cohort):
    from magenta.cost.cache import SemanticCache
    from magenta.cost.meter import CostMeter
    from magenta.graph.batch_diagnose import diagnose_cohort
    from tests.db_fixtures import TENANT_A

    # small_cohort fixture: 2 customers with identical driver signatures -> 1 LLM call, 1 cache hit
    calls = []
    monkeypatch.setattr(
        "magenta.graph.batch_diagnose._chat",
        lambda role, msgs: (calls.append(role), "BILL_SHOCK")[1],
    )
    meter = CostMeter()
    cache = SemanticCache(small_cohort.conn, TENANT_A, small_cohort.embedder, threshold=0.9)
    diagnose_cohort(small_cohort.customers, small_cohort.reports, small_cohort.deps,
                     meter=meter, cache=cache)
    r = meter.report()
    assert r["total_decisions"] == 2 and r["cache_hit_rate"] >= 0.5   # 2nd near-dup hit
    assert len(calls) == 1                                            # only one real LLM call


def test_report_dict_has_required_keys(monkeypatch, small_cohort):
    from magenta.cost.cache import SemanticCache
    from magenta.cost.meter import CostMeter
    from magenta.graph.batch_diagnose import diagnose_cohort
    from tests.db_fixtures import TENANT_A

    monkeypatch.setattr("magenta.graph.batch_diagnose._chat", lambda role, msgs: "BILL_SHOCK")
    meter = CostMeter()
    cache = SemanticCache(small_cohort.conn, TENANT_A, small_cohort.embedder, threshold=0.9)
    diagnose_cohort(small_cohort.customers, small_cohort.reports, small_cohort.deps,
                     meter=meter, cache=cache)
    r = meter.report()
    for key in ("pct_routed_cheap", "cache_hit_rate", "escalation_rate", "total_decisions"):
        assert key in r
