import operator

import pytest
from pydantic import ValidationError

from magenta.brain.risk import Band, Driver
from magenta.brain.uplift import Segment
from magenta.graph.state import (
    Diagnosis,
    GuardrailVerdict,
    OverallState,
    RiskUpliftReport,
    Timing,
)


def test_timing_values():
    assert Timing.ACT_NOW.value == "ACT_NOW"
    assert Timing.SNOOZE.value == "SNOOZE"


def test_risk_uplift_report_roundtrip():
    r = RiskUpliftReport(
        p_churn=0.72,
        band=Band.HIGH,
        drivers=[Driver(feature="OVERAGE_EVENTS", label="Overage events", shap_value=0.31, direction="UP")],
        tau_hat=0.18,
        segment=Segment.PERSUADABLE,
        engage=True,
        timing=Timing.ACT_NOW,
    )
    assert r.engage is True
    assert r.segment is Segment.PERSUADABLE
    # pydantic v2 dump is JSON-safe (enums -> values) => no hidden objects
    dumped = r.model_dump(mode="json")
    assert dumped["timing"] == "ACT_NOW"


def test_diagnosis_defaults_and_confidence_bounds():
    d = Diagnosis(
        root_cause_tags=["BILL_SHOCK", "PRICE_SENSITIVITY"],
        narrative="Customer hit overage twice; margin under pressure.",
        eligible_offer_ids=["BILL_CREDIT", "PLAN_DOWNSELL"],
        confidence=0.83,
    )
    assert 0.0 <= d.confidence <= 1.0
    with pytest.raises(ValidationError):
        Diagnosis(root_cause_tags=[], narrative="x", eligible_offer_ids=[], confidence=1.5)


def test_guardrail_verdict_shape():
    v = GuardrailVerdict(decision="NEEDS_APPROVAL", failed_policies=["VALUE_CAP"])
    assert v.decision == "NEEDS_APPROVAL"
    assert v.failed_policies == ["VALUE_CAP"]


def test_overall_state_reducers_are_append():
    # audit_log uses operator.add; messages uses langgraph add_messages
    ann = OverallState.__annotations__["audit_log"]
    # Annotated[list[dict], operator.add] -> metadata contains operator.add
    assert operator.add in getattr(ann, "__metadata__", ())
