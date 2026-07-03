from magenta.brain.uplift import UpliftModel
from magenta.brain.training import build_training_data


def _fitted():
    td = build_training_data(n=6000, seed=31)
    m = UpliftModel().fit(td.customers, td.treated, td.retained)
    return m, td


def test_tau_is_float():
    m, td = _fitted()
    t = m.tau(td.customers[0])
    assert isinstance(t, float)


def test_qini_positive_on_sim_data():
    m, td = _fitted()
    q = m.qini(td.customers, td.treated, td.retained)
    assert q > 0.0, f"Qini {q} should be positive"


def test_save_load_roundtrip(tmp_path):
    m, td = _fitted()
    p = tmp_path / "uplift.joblib"
    m.save(p)
    m2 = UpliftModel.load(p)
    assert abs(m.tau(td.customers[0]) - m2.tau(td.customers[0])) < 1e-9
