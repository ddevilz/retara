import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


def _featurize(customers: list[Customer]) -> np.ndarray:
    """Numeric/boolean Customer fields, flattened for a plain LogisticRegression.

    Missing NPS is imputed to -1 (a sentinel outside the valid 0-10 range) —
    missingness itself is drawn independently of hidden state, so this can't
    leak segment signal on its own.
    """
    rows = []
    for c in customers:
        rows.append([
            c.tenure_months,
            c.monthly_charge,
            c.total_charges,
            c.data_gb_used_p50,
            c.data_allowance_gb,
            c.overage_events_90d,
            c.dropped_calls_30d,
            c.support_tickets_90d,
            -1.0 if c.nps_last is None else float(c.nps_last),
            c.late_payments_12m,
            c.device_age_months,
            c.contract_end_days,
            c.gross_margin_monthly,
            c.clv_estimate,
            float(c.senior_citizen),
            float(c.has_partner),
            float(c.has_dependents),
        ])
    return np.array(rows, dtype=float)


def test_persuadable_segment_weakly_recoverable_from_observables():
    """Anti-circularity balance check (spec: assign segment PROBABILISTICALLY
    FROM HIDDEN STATE). Hidden thetas already correlate noisily with
    observables, so persuadable_segment must inherit a *weak* observable
    signal by coupling through them — enough that an uplift model targeting
    on X can beat random, not so much that segment is trivially readable off
    X (which would make the sim unrealistically easy / leak-y).

    Locks in both directions: AUC > 0.56 (real signal exists) and < 0.85
    (not trivially recoverable).
    """
    # n=20000: at n=8000 the band floor had only ~1-std margin (review measured
    # 1/20 seeds below 0.56); at 20k, 0/15 seeds dipped below. Keeps the test
    # robust if the seed or featurization ever changes.
    pop, hidden = generate_population(20000, seed=20260702)
    x = _featurize(pop)
    y = np.array(
        [hidden[c.customer_id].persuadable_segment == Segment.PERSUADABLE for c in pop],
        dtype=int,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(x_train, y_train)
    probs = clf.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    assert auc > 0.56, f"segment carries no observable signal (AUC={auc:.3f}) — check theta coupling"
    assert auc < 0.85, f"segment trivially recoverable from observables (AUC={auc:.3f}) — coupling too strong"
