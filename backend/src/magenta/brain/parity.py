"""Real-data parity check: IBM Telco Customer Churn CSV vs Magenta's simulator.

Credibility artifact, not a modeling exercise: trains the SAME pipeline shape
as ``magenta.brain.risk.RiskModel`` (LightGBM base learner -> isotonic
calibration via ``FrozenEstimator`` on a held-out split -> AUC/Brier/ECE on a
fully separate test set) on the real IBM Telco dataset, and prints its
metrics next to a fresh in-memory evaluation of the simulator using the
*actual* ``RiskModel`` class. The claim under test: "same pipeline, comparable
difficulty, sim slightly harder" -- whatever the numbers say, honestly.

Entry points:
    uv run python backend/scripts/real_data_parity.py   (thin wrapper -> main())
    uv run magenta parity                                (CLI subcommand)

Network: ``download_telco_csv`` fetches the CSV ONCE into
``data/telco_real.csv`` (skipped if the file already exists). Every other
function here (feature building, model fit/eval) is pure/offline, so
``tests/test_parity.py`` can exercise ``build_real_features`` with an inline
CSV fixture and never touches the network -- consistent with CLAUDE.md's "no
network in tests" rule (this module is a script, not a test).

Anti-circularity note: this is a *credibility* check, not part of the agent's
decision path -- the real Telco CSV never feeds the simulator or the agent's
runtime features. It exists purely to argue the simulator's churn-prediction
difficulty is realistic.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from magenta.brain.risk import RISK_LGBM_PARAMS, RiskModel, _expected_calibration_error
from magenta.brain.training import build_training_data
from magenta.config import data_dir

SEED = 7
EXPECTED_ROWS = 7043

# Two independent public mirrors of the IBM Telco Customer Churn CSV (same
# 7043-row dataset, verified byte-identical schema). Try PRIMARY first; fall
# back to ALTERNATE if it 404s / times out / the host is unreachable.
PRIMARY_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
ALTERNATE_URL = (
    "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/"
    "master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# Real-column feature spec (documented, reasonable subset of the 19 predictor
# columns in the source CSV). Deliberately excluded: customerID (identifier,
# not a feature) and gender (avoid using a demographic attribute as a
# churn-risk predictor -- same "don't cheat with the wrong kind of signal"
# spirit as magenta's observable-only feature set, just a different axis).
NUMERIC_COLS: list[str] = ["tenure", "MonthlyCharges", "TotalCharges"]
ONEHOT_COLS: list[str] = ["Contract", "InternetService", "PaymentMethod"]
# Yes/No columns (source also uses "No internet service" / "No phone service"
# for some of these -- both fold to 0/False, same as plain "No").
BOOL_YESNO_COLS: list[str] = [
    "Partner",
    "Dependents",
    "PaperlessBilling",
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
# Fixed one-hot column order so a tiny fixture (which won't see every
# category) still produces a matrix with this exact shape/column contract.
_ONEHOT_EXPANDED_COLS: list[str] = [
    "Contract_Month-to-month",
    "Contract_One year",
    "Contract_Two year",
    "InternetService_DSL",
    "InternetService_Fiber optic",
    "InternetService_No",
    "PaymentMethod_Bank transfer (automatic)",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
]
FEATURE_NAMES: list[str] = (
    NUMERIC_COLS + BOOL_YESNO_COLS + ["SeniorCitizen"] + _ONEHOT_EXPANDED_COLS
)


def download_telco_csv(dest: Path) -> Path:
    """Download the IBM Telco CSV into ``dest`` if it doesn't already exist.

    Verifies the result parses to ``EXPECTED_ROWS`` rows with a Churn column
    of Yes/No before accepting it. Tries ``PRIMARY_URL`` then
    ``ALTERNATE_URL``; exits with a clear message if both fail.
    """
    if dest.exists():
        print(f"[parity] found existing {dest}, skipping download")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in (PRIMARY_URL, ALTERNATE_URL):
        try:
            print(f"[parity] downloading {url} ...")
            req = Request(url, headers={"User-Agent": "magenta-parity-check/1.0"})
            with urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed https mirrors
                raw = resp.read()
            dest.write_bytes(raw)
            df = pd.read_csv(dest)
            if df.shape[0] != EXPECTED_ROWS:
                raise ValueError(f"expected {EXPECTED_ROWS} rows, got {df.shape[0]}")
            if "Churn" not in df.columns:
                raise ValueError("missing 'Churn' column")
            bad_labels = set(df["Churn"].unique()) - {"Yes", "No"}
            if bad_labels:
                raise ValueError(f"unexpected Churn values: {bad_labels}")
            print(f"[parity] downloaded + verified: {df.shape[0]} rows, Churn in Yes/No")
            return dest
        except (URLError, HTTPError, ValueError, OSError) as exc:
            print(f"[parity] mirror failed ({url}): {exc}", file=sys.stderr)
            dest.unlink(missing_ok=True)
            last_error = exc

    print(
        "[parity] ERROR: both mirrors failed to produce a valid "
        f"{EXPECTED_ROWS}-row Telco CSV. Last error: {last_error}\n"
        "Place a valid copy at "
        f"{dest} manually and re-run.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def build_real_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build (X, y, feature_names) from the raw IBM Telco columns.

    - numeric: tenure, MonthlyCharges, TotalCharges. ``TotalCharges`` ships as
      a string with 11 blank values in the real 7043-row file -- all of them
      brand-new tenure==0 accounts -- so blanks coerce to 0.0 (the correct
      "no charges accrued yet" value, not an imputation hack).
    - one-hot: Contract, InternetService, PaymentMethod (fixed column order,
      see ``_ONEHOT_EXPANDED_COLS`` -- stable even if a slice of data doesn't
      contain every category, e.g. this file's tiny test fixture).
    - boolean (Yes/No, "No xxx service" folds to No): Partner, Dependents,
      PaperlessBilling, PhoneService, MultipleLines, OnlineSecurity,
      OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
      StreamingMovies.
    - SeniorCitizen: already 0/1 in the source, used as-is.

    label: Churn Yes/No -> 1/0.
    """
    work = df.copy()
    work["TotalCharges"] = pd.to_numeric(work["TotalCharges"], errors="coerce").fillna(0.0)

    numeric = work[NUMERIC_COLS].astype(float).reset_index(drop=True)

    bool_frame = pd.DataFrame(
        {col: (work[col] == "Yes").astype(float) for col in BOOL_YESNO_COLS}
    ).reset_index(drop=True)
    bool_frame["SeniorCitizen"] = work["SeniorCitizen"].astype(float).reset_index(drop=True)

    onehot = pd.get_dummies(work[ONEHOT_COLS], columns=ONEHOT_COLS, dtype=float)
    for col in _ONEHOT_EXPANDED_COLS:
        if col not in onehot.columns:
            onehot[col] = 0.0
    onehot = onehot[_ONEHOT_EXPANDED_COLS].reset_index(drop=True)

    features = pd.concat([numeric, bool_frame, onehot], axis=1)[FEATURE_NAMES]
    X = features.to_numpy(dtype=np.float64)
    y = (work["Churn"].to_numpy() == "Yes").astype(int)
    return X, y, list(features.columns)


@dataclass
class ParityMetrics:
    n_train: int
    n_test: int
    auc: float
    brier: float
    ece: float


def fit_eval_lgbm_isotonic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    calib_size: float = 0.3,
    seed: int = SEED,
) -> ParityMetrics:
    """Mirror ``RiskModel.fit`` + ``evaluate`` on raw feature matrices.

    Same shape as the risk model: LightGBM base learner (``RISK_LGBM_PARAMS``,
    imported from ``magenta.brain.risk`` so hyperparams can never drift out
    of sync) trained on a train sub-split, isotonic-calibrated via
    ``FrozenEstimator`` on the remaining calibration sub-split, evaluated on
    a fully separate ``X_test``/``y_test``. ECE uses the exact same helper
    ``RiskModel.evaluate`` uses, imported directly for bit-identical math.
    """
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_train, y_train, test_size=calib_size, random_state=seed, stratify=y_train
    )
    base = LGBMClassifier(**RISK_LGBM_PARAMS)
    base.fit(X_tr, y_tr)
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    calibrated.fit(X_cal, y_cal)
    p = calibrated.predict_proba(X_test)[:, 1]
    return ParityMetrics(
        n_train=len(y_train),
        n_test=len(y_test),
        auc=float(roc_auc_score(y_test, p)),
        brier=float(brier_score_loss(y_test, p)),
        ece=_expected_calibration_error(np.asarray(y_test), p),
    )


def run_real_parity(csv_path: Path, *, seed: int = SEED) -> ParityMetrics:
    """Real-data AUC/Brier/ECE: 75/25 train/test split, seed=7."""
    df = pd.read_csv(csv_path)
    X, y, _ = build_real_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    return fit_eval_lgbm_isotonic(X_train, y_train, X_test, y_test, seed=seed)


def run_sim_parity(*, n: int = 6000, seed: int = SEED) -> ParityMetrics:
    """Fresh sim-side AUC/Brier/ECE, symmetric with ``run_real_parity``.

    ``build_training_data`` and ``RiskModel`` are pure in-memory (no sqlite
    writes) -- this does NOT touch ``data/*.db``, which a live cohort process
    owns. Same 75/25 train/test split, same seed=7, so the comparison to the
    real-data run is apples-to-apples: one population draw, split once, the
    real ``RiskModel`` class fit on train and evaluated on the held-out test
    slice (``RiskModel.fit`` internally re-splits train into its own
    train/calibration sub-split, exactly mirroring ``fit_eval_lgbm_isotonic``
    above -- it IS the same pipeline, not a lookalike).
    """
    td = build_training_data(n=n, seed=seed)
    y = np.asarray([int(b) for b in td.churned])
    idx = np.arange(len(td.customers))
    idx_train, idx_test = train_test_split(idx, test_size=0.25, random_state=seed, stratify=y)

    train_customers = [td.customers[i] for i in idx_train]
    train_churned = [bool(td.churned[i]) for i in idx_train]
    test_customers = [td.customers[i] for i in idx_test]
    test_churned = [bool(td.churned[i]) for i in idx_test]

    model = RiskModel().fit(train_customers, train_churned)
    report = model.evaluate(test_customers, test_churned)
    return ParityMetrics(
        n_train=len(train_customers),
        n_test=len(test_customers),
        auc=report.auc,
        brier=report.brier,
        ece=report.ece,
    )


def _interpret(real: ParityMetrics, sim: ParityMetrics) -> str:
    gap = real.auc - sim.auc
    if gap > 0.02:
        verdict = (
            f"sim is HARDER than real data (AUC gap {gap:+.3f}) -- consistent with "
            "the claim that Magenta's simulator is at least as difficult to predict "
            "as real telecom churn."
        )
    elif gap < -0.02:
        verdict = (
            f"sim is EASIER than real data (AUC gap {gap:+.3f}) -- reported honestly; "
            "this cuts against the 'comparable/harder difficulty' claim and is worth "
            "investigating (e.g. feature richness, label noise) before citing it."
        )
    else:
        verdict = f"sim and real data are comparably difficult (AUC gap {gap:+.3f})."
    return f"INTERPRETATION: {verdict}"


def format_report(real: ParityMetrics, sim: ParityMetrics) -> str:
    header = f"{'DATASET':<10}{'N_TRAIN':>10}{'N_TEST':>9}{'AUC':>9}{'BRIER':>9}{'ECE':>9}"
    lines = [
        "PARITY REPORT -- real IBM Telco data vs Magenta simulator",
        "Same pipeline: LightGBM (RISK_LGBM_PARAMS) -> isotonic calibration "
        "(FrozenEstimator) -> held-out AUC/Brier/ECE. seed=7, 75/25 train/test split.",
        "",
        header,
        "-" * len(header),
        f"{'real':<10}{real.n_train:>10}{real.n_test:>9}{real.auc:>9.4f}"
        f"{real.brier:>9.4f}{real.ece:>9.4f}",
        f"{'sim':<10}{sim.n_train:>10}{sim.n_test:>9}{sim.auc:>9.4f}"
        f"{sim.brier:>9.4f}{sim.ece:>9.4f}",
        "",
        _interpret(real, sim),
    ]
    return "\n".join(lines)


def main() -> None:
    csv_path = data_dir() / "telco_real.csv"
    download_telco_csv(csv_path)

    real = run_real_parity(csv_path)
    sim = run_sim_parity()
    report = format_report(real, sim)

    print()
    print(report)

    out_path = data_dir() / "parity_report.txt"
    out_path.write_text(report + "\n", encoding="utf-8")
    print(f"\n[parity] wrote {out_path}")


if __name__ == "__main__":
    main()
