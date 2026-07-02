import json
import math

from magenta.config import data_dir


def _load():
    with (data_dir() / "telco_marginals.json").open() as fh:
        return json.load(fh)


def test_categorical_distributions_sum_to_one():
    m = _load()
    for key in ("contract", "plan", "segment_mix"):
        total = sum(m[key].values())
        assert math.isclose(total, 1.0, abs_tol=1e-6), (key, total)


def test_segment_mix_matches_spec_targets():
    seg = _load()["segment_mix"]
    assert math.isclose(seg["PERSUADABLE"], 0.25, abs_tol=0.02)
    assert math.isclose(seg["SURE_THING"], 0.50, abs_tol=0.02)
    assert math.isclose(seg["LOST_CAUSE"], 0.17, abs_tol=0.02)
    assert math.isclose(seg["SLEEPING_DOG"], 0.08, abs_tol=0.02)


def test_plausible_rates():
    m = _load()
    assert 0.0 < m["nps_missing_rate"] < 1.0
    assert 0.1 < m["base_churn_rate"] < 0.5
