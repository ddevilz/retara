"""Graph state + LLM/ML-contract artifacts for the LangGraph decision spine.

Spec §6. All enum values ALL_CAPS. Pydantic v2. `messages` + `audit_log`
use append reducers so parallel/streamed writes accumulate instead of clobber.

CRITICAL: no L1 latent field (theta_churn, theta_price, persuadable_segment,
competitor_pull) ever appears here. This TypedDict is the agent's entire world.
"""
import operator
from enum import Enum
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator

from magenta.brain.risk import Band, Driver
from magenta.brain.uplift import Segment
from magenta.offers import Arm, OfferDecision


class Timing(str, Enum):
    ACT_NOW = "ACT_NOW"
    SNOOZE = "SNOOZE"


class RiskUpliftReport(BaseModel):
    """ML brain output (§5.1). Narrated by the diagnose LLM; never re-derived."""

    p_churn: float = Field(ge=0.0, le=1.0)
    band: Band
    drivers: list[Driver]
    tau_hat: float
    segment: Segment
    engage: bool
    timing: Timing


class Diagnosis(BaseModel):
    """Cheap-role LLM output. Narrates SHAP drivers; constrains the arm set."""

    root_cause_tags: list[str]
    narrative: str
    eligible_offer_ids: list[str]
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("eligible_offer_ids")
    @classmethod
    def _only_real_arms(cls, ids: list[str]) -> list[str]:
        # LLM output boundary: silently drop hallucinated arm names so the
        # audit trail can distinguish "no eligible offers" (empty AFTER a
        # non-empty raw list) from genuine ineligibility. Valid ids pass through.
        return [i for i in ids if i in Arm._value2member_map_]


class GuardrailVerdict(BaseModel):
    """Deterministic compliance node output (§5.7). Fails closed."""

    decision: str  # PASS | REJECT | NEEDS_APPROVAL
    failed_policies: list[str] = Field(default_factory=list)


class OverallState(TypedDict):
    customer_id: str
    campaign_id: str
    consent_flags: dict
    risk: RiskUpliftReport | None
    diagnosis: Diagnosis | None
    offer: OfferDecision | None
    verdict: GuardrailVerdict | None
    fulfillment: dict | None
    outcome: dict | None
    messages: Annotated[list, add_messages]
    audit_log: Annotated[list[dict], operator.add]
    requires_approval: bool
    holdout: bool
