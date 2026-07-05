"""Offline test for the real-data feature builder (magenta.brain.parity).

NO network access here: build_real_features() is a pure pandas/numpy
transform over an inline CSV fixture shaped like the real IBM Telco file's
header, exercising blank-TotalCharges coercion, Yes/No + "No xxx service"
boolean folding, and one-hot column stability.

The fixture is built from explicit column/row lists (not a hand-typed CSV
blob) to avoid silent column-count drift, then serialized to real CSV text
via ``to_csv``/``read_csv`` so the blank TotalCharges case is exercised as
actual CSV text, not a pre-built NaN.
"""
from __future__ import annotations

import io

import pandas as pd

from magenta.brain.parity import FEATURE_NAMES, build_real_features

_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]

_ROWS = [
    # blank TotalCharges (brand-new, tenure=1), Churn=No
    ["7590-VHVEG", "Female", 0, "Yes", "No", 1, "No", "No phone service", "DSL",
     "No", "Yes", "No", "No", "No", "No", "Month-to-month", "Yes",
     "Electronic check", 29.85, "", "No"],
    # senior citizen, Fiber optic, Churn=Yes
    ["5575-GNVDE", "Male", 1, "No", "No", 34, "Yes", "No", "Fiber optic",
     "Yes", "No", "Yes", "No", "Yes", "No", "One year", "No",
     "Mailed check", 56.95, 1889.5, "Yes"],
    # no internet service at all -> all internet-addon cols fold to "No"
    ["3668-QPYBK", "Female", 0, "No", "Yes", 2, "Yes", "Yes", "No",
     "No internet service", "No internet service", "No internet service",
     "No internet service", "No internet service", "No internet service",
     "Two year", "Yes", "Bank transfer (automatic)", 20.05, 40.10, "No"],
    # everything Yes-heavy, Churn=Yes
    ["9237-HQITU", "Male", 0, "Yes", "Yes", 45, "Yes", "No", "DSL",
     "Yes", "Yes", "Yes", "Yes", "No", "No", "Month-to-month", "No",
     "Credit card (automatic)", 42.30, 1903.5, "Yes"],
]


def _fixture_df() -> pd.DataFrame:
    raw = pd.DataFrame(_ROWS, columns=_COLUMNS)
    csv_text = raw.to_csv(index=False)
    return pd.read_csv(io.StringIO(csv_text))


def test_matrix_shape_matches_feature_names():
    X, y, feature_names = build_real_features(_fixture_df())
    assert feature_names == FEATURE_NAMES
    assert X.shape == (4, len(FEATURE_NAMES))
    assert y.shape == (4,)


def test_label_encoding_churn_yes_no():
    _, y, _ = build_real_features(_fixture_df())
    assert y.tolist() == [0, 1, 0, 1]


def test_blank_total_charges_coerced_to_zero():
    X, _, feature_names = build_real_features(_fixture_df())
    col = feature_names.index("TotalCharges")
    assert X[0, col] == 0.0
    assert X[1, col] == 1889.5


def test_boolean_yesno_and_no_service_folding():
    X, _, feature_names = build_real_features(_fixture_df())
    partner_col = feature_names.index("Partner")
    online_security_col = feature_names.index("OnlineSecurity")
    senior_col = feature_names.index("SeniorCitizen")

    # row 0: Partner=Yes -> 1.0
    assert X[0, partner_col] == 1.0
    # row 1: Partner=No -> 0.0; SeniorCitizen=1 -> 1.0
    assert X[1, partner_col] == 0.0
    assert X[1, senior_col] == 1.0
    # row 2: OnlineSecurity="No internet service" folds to 0.0, same as "No"
    assert X[2, online_security_col] == 0.0


def test_onehot_columns_stable_and_correctly_encoded():
    X, _, feature_names = build_real_features(_fixture_df())
    for expected_col in (
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
    ):
        assert expected_col in feature_names

    # row 2 (3668-QPYBK) has InternetService=No -> that one-hot column is 1.0.
    row2_internet_no = feature_names.index("InternetService_No")
    assert X[2, row2_internet_no] == 1.0
