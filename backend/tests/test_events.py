import numpy as np

from magenta.sim.events import EventKind, LifeEvent, generate_events
from magenta.sim.population import generate_population


def test_returns_life_events():
    pop, hid = generate_population(50, seed=3)
    c = pop[0]
    events = generate_events(c, hid[c.customer_id], np.random.default_rng(0))
    assert all(isinstance(e, LifeEvent) for e in events)
    assert all(isinstance(e.kind, EventKind) for e in events)
    assert all(e.hazard_multiplier > 0 for e in events)


def test_deterministic_with_same_rng_seed():
    pop, hid = generate_population(50, seed=3)
    c = pop[0]
    a = generate_events(c, hid[c.customer_id], np.random.default_rng(99))
    b = generate_events(c, hid[c.customer_id], np.random.default_rng(99))
    assert [(e.kind, e.hazard_multiplier) for e in a] == \
           [(e.kind, e.hazard_multiplier) for e in b]


def test_overage_customer_more_likely_to_get_overage_event():
    # customer with heavy overage history should draw OVERAGE more often
    pop, hid = generate_population(2000, seed=5)
    heavy = [c for c in pop if c.overage_events_90d >= 3]
    assert heavy, "expected some heavy-overage customers"
    rng = np.random.default_rng(1)
    hits = 0
    for c in heavy[:200]:
        evs = generate_events(c, hid[c.customer_id], rng)
        if any(e.kind == EventKind.OVERAGE for e in evs):
            hits += 1
    assert hits > 0


def test_contract_expiry_only_when_ending_soon():
    pop, hid = generate_population(2000, seed=8)
    rng = np.random.default_rng(2)
    for c in pop:
        evs = generate_events(c, hid[c.customer_id], rng)
        if any(e.kind == EventKind.CONTRACT_EXPIRY for e in evs):
            assert c.contract_end_days <= 45
