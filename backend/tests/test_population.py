import json

from magenta.sim.population import (
    Customer,
    HiddenState,
    Segment,
    generate_population,
)


def test_reproducible_same_seed():
    pop_a, hid_a = generate_population(500, seed=42)
    pop_b, hid_b = generate_population(500, seed=42)
    assert [c.model_dump() for c in pop_a] == [c.model_dump() for c in pop_b]
    assert {k: v.model_dump() for k, v in hid_a.items()} == {
        k: v.model_dump() for k, v in hid_b.items()
    }


def test_different_seed_differs():
    pop_a, _ = generate_population(500, seed=1)
    pop_b, _ = generate_population(500, seed=2)
    assert [c.customer_id for c in pop_a] != []
    assert [c.monthly_charge for c in pop_a] != [c.monthly_charge for c in pop_b]


def test_no_hidden_field_on_customer_model():
    # schema-level: observable and hidden field sets are disjoint
    cust_fields = set(Customer.model_fields)
    hidden_fields = set(HiddenState.model_fields)
    assert cust_fields & hidden_fields == set()
    for banned in ("theta_churn_base", "theta_price_sens",
                   "persuadable_segment", "competitor_pull"):
        assert banned not in cust_fields


def test_no_hidden_key_in_serialized_population():
    pop, _ = generate_population(300, seed=7)
    blob = json.dumps([c.model_dump() for c in pop])
    for banned in ("theta_", "persuadable_segment", "competitor_pull"):
        assert banned not in blob


def test_segment_mix_approximately_matches_targets():
    _, hidden = generate_population(20000, seed=123)
    counts = {s: 0 for s in Segment}
    for hs in hidden.values():
        counts[hs.persuadable_segment] += 1
    n = len(hidden)
    assert abs(counts[Segment.PERSUADABLE] / n - 0.25) < 0.03
    assert abs(counts[Segment.SURE_THING] / n - 0.50) < 0.03
    assert abs(counts[Segment.LOST_CAUSE] / n - 0.17) < 0.03
    assert abs(counts[Segment.SLEEPING_DOG] / n - 0.08) < 0.03


def test_nps_missing_roughly_40pct():
    pop, _ = generate_population(5000, seed=9)
    missing = sum(1 for c in pop if c.nps_last is None)
    assert 0.34 < missing / len(pop) < 0.46


def test_hidden_store_keyed_by_customer_id():
    pop, hidden = generate_population(200, seed=11)
    assert set(hidden) == {c.customer_id for c in pop}
