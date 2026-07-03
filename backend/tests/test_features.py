import numpy as np
from magenta.brain.features import featurize, FEATURE_NAMES
from magenta.sim.population import generate_population


HIDDEN_FIELDS = {"theta_churn_base", "theta_price_sens", "persuadable_segment", "competitor_pull"}


def _one_customer():
    customers, _hidden = generate_population(50, seed=1)
    return customers[0]


def test_featurize_length_matches_names():
    x = featurize(_one_customer())
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float64
    assert x.shape == (len(FEATURE_NAMES),)


def test_no_hidden_field_leaks_into_features():
    lowered = {name.lower() for name in FEATURE_NAMES}
    assert lowered.isdisjoint(HIDDEN_FIELDS)


def test_deterministic_order():
    c = _one_customer()
    a = featurize(c)
    b = featurize(c)
    assert np.array_equal(a, b)


def test_missing_nps_imputes_sentinel():
    customers, _ = generate_population(200, seed=3)
    target = next(c for c in customers if c.nps_last is None)
    x = featurize(target)
    idx = FEATURE_NAMES.index("NPS_LAST")
    assert x[idx] == -999.0
