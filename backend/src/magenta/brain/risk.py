"""Calibrated churn-risk model: LightGBM + isotonic calibration + TreeSHAP drivers.

Trains on observable features + oracle churn labels ONLY (anti-circularity).
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path

import joblib
import numpy as np
import shap
from lightgbm import LGBMClassifier
from pydantic import BaseModel
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from magenta.config import data_dir
from magenta.brain.features import FEATURE_NAMES, featurize
from magenta.sim.population import Customer

_DEFAULT_PATH = data_dir() / "models" / "risk.joblib"

# Human-readable labels for driver narration (LLM narrates these, never re-derives).
_FEATURE_LABELS: dict[str, str] = {
    "TENURE_MONTHS": "tenure (months)",
    "MONTHLY_CHARGE": "monthly charge",
    "TOTAL_CHARGES": "total charges to date",
    "DATA_GB_USED_P50": "typical monthly data use",
    "DATA_ALLOWANCE_GB": "data allowance",
    "OVERAGE_EVENTS_90D": "data overage events (90d)",
    "DROPPED_CALLS_30D": "dropped calls (30d)",
    "SUPPORT_TICKETS_90D": "support tickets (90d)",
    "LATE_PAYMENTS_12M": "late payments (12m)",
    "DEVICE_AGE_MONTHS": "device age (months)",
    "CONTRACT_END_DAYS": "days to contract end",
    "GROSS_MARGIN_MONTHLY": "monthly gross margin",
    "CLV_ESTIMATE": "customer lifetime value",
    "NPS_LAST": "last NPS score",
    "DATA_UTIL_RATIO": "data utilisation ratio",
    "AVG_CHARGE_PER_TENURE": "avg charge per tenure month",
}


class Band(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Driver(BaseModel):
    feature: str
    label: str
    shap_value: float
    direction: str  # "UP" (raises churn) | "DOWN" (lowers churn)


class RiskAssessment(BaseModel):
    p_churn: float
    band: Band
    drivers: list[Driver]


class RiskEvalReport(BaseModel):
    auc: float
    brier: float
    ece: float


def _band_for(p: float) -> Band:
    if p < 0.25:
        return Band.LOW
    if p < 0.50:
        return Band.MEDIUM
    if p < 0.75:
        return Band.HIGH
    return Band.CRITICAL


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not mask.any():
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


class RiskModel:
    def __init__(self) -> None:
        self._calibrated: CalibratedClassifierCV | None = None
        self._raw: LGBMClassifier | None = None  # uncalibrated base for TreeSHAP
        self.feature_names: list[str] = list(FEATURE_NAMES)

    def _matrix(self, customers: list[Customer]) -> np.ndarray:
        return np.vstack([featurize(c) for c in customers])

    def fit(self, customers: list[Customer], churned: list[bool]) -> "RiskModel":
        X = self._matrix(customers)
        y = np.asarray([int(b) for b in churned])
        X_tr, X_cal, y_tr, y_cal = train_test_split(
            X, y, test_size=0.3, random_state=0, stratify=y
        )
        base = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=0,
            verbose=-1,
        )
        base.fit(X_tr, y_tr)
        self._raw = base
        # Isotonic calibration on the held-out split. `cv="prefit"` was removed
        # from CalibratedClassifierCV (sklearn>=1.6); FrozenEstimator is the
        # replacement for wrapping an already-fitted estimator.
        self._calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
        self._calibrated.fit(X_cal, y_cal)
        self._explainer = None  # invalidate cached TreeExplainer (refit => new trees)
        return self

    def _p_churn(self, X: np.ndarray) -> np.ndarray:
        assert self._calibrated is not None, "model not fitted"
        return self._calibrated.predict_proba(X)[:, 1]

    def score(self, c: Customer) -> RiskAssessment:
        assert self._raw is not None, "model not fitted"
        x = featurize(c).reshape(1, -1)
        p = float(self._p_churn(x)[0])
        drivers = self._top_drivers(x)
        return RiskAssessment(p_churn=p, band=_band_for(p), drivers=drivers)

    def p_churn_batch(self, customers: list[Customer]) -> np.ndarray:
        """Vectorized calibrated churn probabilities — no SHAP (use for cohorts;
        per-customer TreeSHAP in score() is ~100x slower and only needed for drivers)."""
        return self._p_churn(self._matrix(customers))

    def _top_drivers(self, x: np.ndarray, k: int = 5) -> list[Driver]:
        # Cache the TreeExplainer: rebuilding it per score() call was ~100x the
        # cost of the shap_values computation itself (measured ~13min/rung at
        # n=10k in the ablation ladder before this cache).
        explainer = getattr(self, "_explainer", None)
        if explainer is None:
            explainer = shap.TreeExplainer(self._raw)
            self._explainer = explainer
        sv = explainer.shap_values(x)
        # LightGBM binary may return a list [class0, class1]; take positive class.
        if isinstance(sv, list):
            sv = sv[1]
        row = np.asarray(sv)[0]
        order = np.argsort(np.abs(row))[::-1][:k]
        out: list[Driver] = []
        for idx in order:
            name = self.feature_names[idx]
            val = float(row[idx])
            out.append(
                Driver(
                    feature=name,
                    label=_FEATURE_LABELS.get(name, name),
                    shap_value=val,
                    direction="UP" if val >= 0 else "DOWN",
                )
            )
        return out

    def evaluate(self, customers: list[Customer], churned: list[bool]) -> RiskEvalReport:
        X = self._matrix(customers)
        y = np.asarray([int(b) for b in churned])
        p = self._p_churn(X)
        return RiskEvalReport(
            auc=float(roc_auc_score(y, p)),
            brier=float(brier_score_loss(y, p)),
            ece=_expected_calibration_error(y, p),
        )

    def save(self, path: str | Path = _DEFAULT_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"calibrated": self._calibrated, "raw": self._raw, "features": self.feature_names},
            path,
        )

    @classmethod
    def load(cls, path: str | Path = _DEFAULT_PATH) -> "RiskModel":
        blob = joblib.load(Path(path))
        m = cls()
        m._calibrated = blob["calibrated"]
        m._raw = blob["raw"]
        m.feature_names = blob["features"]
        return m
