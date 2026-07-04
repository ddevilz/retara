"""Thompson-sampling Bayesian linear contextual bandit over offer arms.

Per arm a: Gaussian posterior on theta_a with precision A_a and b_a where
  A_a = lam*I + sum x x^T,   b_a = sum r*x,   mean = A_a^-1 b_a,  cov = sigma2 * A_a^-1.
select() samples theta_a ~ N(mean, cov) per eligible arm, argmax of x·theta.
Posteriors persisted in SQLite table BANDIT_POSTERIOR (ALL_CAPS columns).
Reward = retained * gross_margin_monthly * 12 - offer.cost (set by caller).
"""
from __future__ import annotations

import numpy as np

from magenta.offers import Arm

_TABLE = "BANDIT_POSTERIOR"


class ThompsonBandit:
    def __init__(
        self,
        dim: int,
        arms: list[Arm],
        lam: float = 1.0,
        sigma2: float = 0.25,
        seed: int = 0,
    ) -> None:
        self.dim = dim
        self.arms = list(arms)
        self.lam = lam
        self.sigma2 = sigma2
        self._A: dict[Arm, np.ndarray] = {a: lam * np.eye(dim) for a in arms}
        self._b: dict[Arm, np.ndarray] = {a: np.zeros(dim) for a in arms}
        self._n: dict[Arm, int] = {a: 0 for a in arms}
        self._rng = np.random.default_rng(seed)

    def _theta_mean(self, arm: Arm) -> np.ndarray:
        return np.linalg.solve(self._A[arm], self._b[arm])

    def posterior_mean(self, x: np.ndarray, arm: Arm) -> float:
        """Expected reward x . theta_mean_a — the deterministic (non-sampled)
        posterior-mean estimate for one arm. Used by System-2's lookahead
        (magenta.graph.system2.deliberate), which must score candidates
        WITHOUT a stochastic Thompson draw so its argmax is reproducible
        within a single deliberation call.
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        return float(x @ self._theta_mean(arm))

    def _sample_theta(self, arm: Arm) -> np.ndarray:
        A_inv = np.linalg.inv(self._A[arm])
        mean = A_inv @ self._b[arm]
        cov = self.sigma2 * A_inv
        return self._rng.multivariate_normal(mean, cov)

    def select(self, x: np.ndarray, eligible: list[Arm]) -> tuple[Arm, float]:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        cand = [a for a in eligible if a in self._A]
        if not cand:
            raise ValueError("no eligible arm known to bandit")
        # One TS draw to choose.
        scores = {a: float(x @ self._sample_theta(a)) for a in cand}
        chosen = max(scores, key=scores.get)
        # MC propensity: fraction of 100 draws where chosen arm is argmax.
        wins = 0
        draws = 100
        for _ in range(draws):
            s = {a: float(x @ self._sample_theta(a)) for a in cand}
            if max(s, key=s.get) == chosen:
                wins += 1
        propensity = max(wins / draws, 1.0 / draws)  # never 0
        return chosen, propensity

    def update(self, x: np.ndarray, arm: Arm, reward: float) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        self._A[arm] = self._A[arm] + np.outer(x, x)  # rank-1
        self._b[arm] = self._b[arm] + reward * x
        self._n[arm] += 1

    def save(self, conn) -> None:
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {_TABLE} ("
            "ARM TEXT PRIMARY KEY, A_MATRIX BLOB, B_VECTOR BLOB, N_UPDATES INTEGER)"
        )
        for a in self.arms:
            conn.execute(
                f"INSERT OR REPLACE INTO {_TABLE} (ARM, A_MATRIX, B_VECTOR, N_UPDATES) "
                "VALUES (?, ?, ?, ?)",
                (
                    a.value,
                    self._A[a].astype(np.float64).tobytes(),
                    self._b[a].astype(np.float64).tobytes(),
                    self._n[a],
                ),
            )
        conn.commit()

    def load(self, conn) -> None:
        cur = conn.execute(f"SELECT ARM, A_MATRIX, B_VECTOR, N_UPDATES FROM {_TABLE}")
        by_value = {a.value: a for a in self.arms}
        for arm_val, a_blob, b_blob, n in cur.fetchall():
            arm = by_value.get(arm_val)
            if arm is None:
                continue
            self._A[arm] = np.frombuffer(a_blob, dtype=np.float64).reshape(self.dim, self.dim).copy()
            self._b[arm] = np.frombuffer(b_blob, dtype=np.float64).reshape(self.dim).copy()
            self._n[arm] = int(n)
