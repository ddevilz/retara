from magenta.cost.meter import CostMeter


def test_meter_percentages():
    m = CostMeter()
    m.record("cheap", cache_hit=True, escalated=False)
    m.record("cheap", cache_hit=False, escalated=False)
    m.record("large", cache_hit=False, escalated=True)
    r = m.report()
    assert r["total_decisions"] == 3
    assert abs(r["cache_hit_rate"] - 1 / 3) < 1e-9
    assert abs(r["pct_routed_cheap"] - 2 / 3) < 1e-9
    assert abs(r["escalation_rate"] - 1 / 3) < 1e-9
