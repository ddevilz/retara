"""Uplift (heterogeneous treatment effect) model: S-learner + T-learner + Qini.

tau(c) = P(retained | treated, x) - P(retained | control, x). T-learner is primary.
Trains on observables + randomized-offer oracle labels ONLY.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklift.metrics import qini_auc_score

from magenta.brain.features import featurize
from magenta.config import data_dir
from magenta.sim.population import Customer

_DEFAULT_PATH = data_dir() / "models" / "uplift.joblib"


def _base() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=0,
        verbose=-1,
    )


class UpliftModel:
    def __init__(self) -> None:
        # T-learner: separate model per treatment arm.
        self._m_treated: LGBMClassifier | None = None
        self._m_control: LGBMClassifier | None = None
        # S-learner: single model with treatment as a feature.
        self._m_single: LGBMClassifier | None = None

    def _matrix(self, customers: list[Customer]) -> np.ndarray:
        return np.vstack([featurize(c) for c in customers])

    def fit(
        self,
        customers: list[Customer],
        treated: list[bool],
        retained: list[bool],
    ) -> "UpliftModel":
        X = self._matrix(customers)
        t = np.asarray([int(b) for b in treated])
        y = np.asarray([int(b) for b in retained])

        # T-learner.
        self._m_treated = _base().fit(X[t == 1], y[t == 1])
        self._m_control = _base().fit(X[t == 0], y[t == 0])

        # S-learner (treatment appended as a feature column).
        Xs = np.hstack([X, t.reshape(-1, 1).astype(float)])
        self._m_single = _base().fit(Xs, y)
        return self

    def _tau_t(self, X: np.ndarray) -> np.ndarray:
        p1 = self._m_treated.predict_proba(X)[:, 1]
        p0 = self._m_control.predict_proba(X)[:, 1]
        return p1 - p0

    def _tau_s(self, X: np.ndarray) -> np.ndarray:
        x1 = np.hstack([X, np.ones((len(X), 1))])
        x0 = np.hstack([X, np.zeros((len(X), 1))])
        return self._m_single.predict_proba(x1)[:, 1] - self._m_single.predict_proba(x0)[:, 1]

    def tau(self, c: Customer) -> float:
        x = featurize(c).reshape(1, -1)
        return float(self._tau_t(x)[0])

    def tau_batch(self, customers: list[Customer]) -> np.ndarray:
        return self._tau_t(self._matrix(customers))

    def qini(
        self,
        customers: list[Customer],
        treated: list[bool],
        retained: list[bool],
    ) -> float:
        uplift = self.tau_batch(customers)
        y = np.asarray([int(b) for b in retained])
        t = np.asarray([int(b) for b in treated])
        return float(qini_auc_score(y_true=y, uplift=uplift, treatment=t))

    def save(self, path: str | Path = _DEFAULT_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"treated": self._m_treated, "control": self._m_control, "single": self._m_single},
            path,
        )

    @classmethod
    def load(cls, path: str | Path = _DEFAULT_PATH) -> "UpliftModel":
        blob = joblib.load(Path(path))
        m = cls()
        m._m_treated = blob["treated"]
        m._m_control = blob["control"]
        m._m_single = blob["single"]
        return m
