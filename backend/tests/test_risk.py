from pathlib import Path

from magenta.brain.features import FEATURE_NAMES
from magenta.brain.risk import Band, RiskModel
from magenta.brain.training import build_training_data


def _fitted_model():
    td = build_training_data(n=6000, seed=21)
    m = RiskModel()
    m.fit(td.customers, td.churned)
    return m, td


def test_auc_in_anticircularity_band():
    m, _ = _fitted_model()
    te = build_training_data(n=3000, seed=222)
    report = m.evaluate(te.customers, te.churned)
    assert 0.72 <= report.auc <= 0.92, f"AUC {report.auc} outside anti-circularity band"


def test_calibration_ece_under_threshold():
    m, _ = _fitted_model()
    te = build_training_data(n=3000, seed=222)
    report = m.evaluate(te.customers, te.churned)
    assert report.ece < 0.08, f"ECE {report.ece} too high"


def test_score_returns_top5_signed_drivers():
    m, td = _fitted_model()
    a = m.score(td.customers[0])
    assert 0.0 <= a.p_churn <= 1.0
    assert isinstance(a.band, Band)
    assert len(a.drivers) == 5
    for d in a.drivers:
        assert d.feature in FEATURE_NAMES
        assert d.direction in {"UP", "DOWN"}


def test_save_load_roundtrip(tmp_path):
    m, td = _fitted_model()
    p = tmp_path / "risk.joblib"
    m.save(p)
    assert Path(p).exists()
    m2 = RiskModel.load(p)
    before = m.score(td.customers[0]).p_churn
    after = m2.score(td.customers[0]).p_churn
    assert abs(before - after) < 1e-9
