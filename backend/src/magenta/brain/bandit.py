"""Thompson-sampling Bayesian linear contextual bandit over offer arms.

Per arm a: Gaussian posterior on theta_a with precision A_a and b_a where
  A_a = lam*I + sum x x^T,   b_a = sum r*x,   mean = A_a^-1 b_a,  cov = sigma2 * A_a^-1.
select() samples theta_a ~ N(mean, cov) per eligible arm, argmax of x·theta.
Posteriors persisted in Postgres table BANDIT_POSTERIOR, tenant-scoped by
TENANT_ID (composite PK with ARM; ALL_CAPS columns, schema owned by Alembic).
Reward = retained * gross_margin_monthly * 12 - offer.cost (set by caller).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sqlalchemy import text
from sqlalchemy.engine import Connection

from magenta.offers import Arm


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

    def _sample_thetas(self, arm: Arm, n: int) -> np.ndarray:
        """n draws of theta_a ~ N(mean, cov), batched in one call.

        method="cholesky" (not the default "svd"): mean/cov are constant
        across all n draws here, so one Cholesky factorization serves all of
        them. select() used to call the SVD-based single-draw form 101x per
        arm (1 chosen draw + 100 MC propensity draws) — ~100x more
        decompositions than necessary, slow enough that a `bandit episodes`
        cohort run took minutes and read as a hang under a 60s test timeout.
        """
        A_inv = np.linalg.inv(self._A[arm])
        mean = A_inv @ self._b[arm]
        cov = self.sigma2 * A_inv
        return self._rng.multivariate_normal(mean, cov, size=n, method="cholesky")

    def select(self, x: np.ndarray, eligible: Sequence[Arm]) -> tuple[Arm, float]:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        cand = [a for a in eligible if a in self._A]
        if not cand:
            raise ValueError("no eligible arm known to bandit")
        draws = 100
        # One batched draw per arm: [0] is the TS draw to choose, [1:] are
        # the MC propensity draws.
        samples = {a: self._sample_thetas(a, 1 + draws) for a in cand}
        scores = {a: float(x @ samples[a][0]) for a in cand}
        chosen = max(scores, key=lambda a: scores[a])
        # MC propensity: fraction of 100 draws where chosen arm is argmax.
        wins = 0
        for i in range(1, draws + 1):
            s = {a: float(x @ samples[a][i]) for a in cand}
            if max(s, key=lambda a: s[a]) == chosen:
                wins += 1
        propensity = max(wins / draws, 1.0 / draws)  # never 0
        return chosen, propensity

    def update(self, x: np.ndarray, arm: Arm, reward: float) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        self._A[arm] = self._A[arm] + np.outer(x, x)  # rank-1
        self._b[arm] = self._b[arm] + reward * x
        self._n[arm] += 1

    def save(self, conn: Connection, tenant_id: str) -> None:
        """Schema lives in Alembic (migration 0001). A_MATRIX/B_VECTOR are raw
        float64 buffers with no embedded dtype — `load` must reshape with the
        same dim and dtype or it reinterprets the bits silently."""
        for a in self.arms:
            conn.execute(
                text(
                    'INSERT INTO "BANDIT_POSTERIOR" '
                    '("TENANT_ID", "ARM", "A_MATRIX", "B_VECTOR", "N_UPDATES") '
                    "VALUES (:tenant_id, :arm, :a, :b, :n) "
                    'ON CONFLICT ("TENANT_ID", "ARM") DO UPDATE SET '
                    '"A_MATRIX" = EXCLUDED."A_MATRIX", '
                    '"B_VECTOR" = EXCLUDED."B_VECTOR", '
                    '"N_UPDATES" = EXCLUDED."N_UPDATES"'
                ),
                {
                    "tenant_id": tenant_id,
                    "arm": a.value,
                    "a": self._A[a].astype(np.float64).tobytes(),
                    "b": self._b[a].astype(np.float64).tobytes(),
                    "n": self._n[a],
                },
            )
        conn.commit()

    def load(self, conn: Connection, tenant_id: str) -> None:
        """Tenant-scoped. A tenant with no saved rows leaves the bandit at its
        prior (no error) — that's the cold-start path every CLI/API call site
        hits on a fresh BANDIT_POSTERIOR table.

        Commits at the end like `save()` does: a bare SELECT still opens a
        transaction, and `api.deps.get_graph_deps()` holds its connection for as
        long as its `DEPS_CACHE` entry lives (a bounded TTL cache, not a
        process-lifetime `@lru_cache`) — an uncommitted read here left that
        connection idle-in-transaction for that whole window, ACCESS-SHARE-locking
        BANDIT_POSTERIOR and blocking any later TRUNCATE on it for the rest
        of the test session."""
        rows = conn.execute(
            text(
                'SELECT "ARM", "A_MATRIX", "B_VECTOR", "N_UPDATES" '
                'FROM "BANDIT_POSTERIOR" WHERE "TENANT_ID" = :tenant_id'
            ),
            {"tenant_id": tenant_id},
        ).mappings().all()
        by_value = {a.value: a for a in self.arms}
        for row in rows:
            arm = by_value.get(row["ARM"])
            if arm is None:
                continue  # an arm retired since this posterior was written
            self._A[arm] = np.frombuffer(
                row["A_MATRIX"], dtype=np.float64
            ).reshape(self.dim, self.dim).copy()
            self._b[arm] = np.frombuffer(
                row["B_VECTOR"], dtype=np.float64
            ).reshape(self.dim).copy()
            self._n[arm] = int(row["N_UPDATES"])
        conn.commit()
