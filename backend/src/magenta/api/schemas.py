"""API request/response schemas — thin mirrors of the magenta package types.

These exist so the OpenAPI doc + frontend types stay honest. They intentionally
duplicate a few package fields rather than re-export pydantic models, because the
API surface should be stable even if internal models are refactored.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "magenta-retain"
    version: str = "0.1.0"


class ScorecardData(BaseModel):
    churn_treatment: float
    churn_holdout: float
    ate: float
    ci_low: float
    ci_high: float
    wasted_offer_rate: float
    sleeping_dogs_contacted: int
    euros_retained: float
    offer_spend: float
    acceptance_rate: float
    n_treatment: int
    n_holdout: int
    offers_made: int


class Rung(BaseModel):
    policy: Literal["noaction", "rules", "risk_rules", "agent_s1", "agent"]
    scorecard: ScorecardData


class Scorecards(BaseModel):
    rungs: list[Rung]


class CustomerSummary(BaseModel):
    """Observable-only projection of a Customer for list views."""

    customer_id: str
    tenure_months: int
    contract: str
    monthly_charges: float
    total_charges: float
    data_util_ratio: float
    dropped_call_rate: float
    nps: float | None = None
    support_tickets: int
    contract_end_days: int
    clv: float
    gross_margin: float


class AuditRow(BaseModel):
    id: int
    ts: str
    customer_id: str
    node: str
    decision: dict
    rationale: str
    holdout: bool


class Customer360(BaseModel):
    customer: CustomerSummary
    audit: list[AuditRow]


# ---- request bodies ----
class RunOneRequest(BaseModel):
    customer_id: str


class ExperimentRequest(BaseModel):
    policy: Literal["noaction", "rules", "risk_rules", "agent_s1", "agent"]
    n: int = 200
    seed: int = 7


class ChatStartRequest(BaseModel):
    mode: Literal["persona", "human"]
    archetype: str | None = None   # required when mode == "persona"
    customer_id: str | None = None


class ChatStartResponse(BaseModel):
    session_id: str
    mode: str
    customer_id: str
    archetype: str | None = None


class ChatTurnRequest(BaseModel):
    text: str


class OrgProfile(BaseModel):
    name: str
    industry: str | None
    monthly_token_budget: int | None
    admin_contact_email: str | None


class OrgProfileUpdate(BaseModel):
    name: str
    industry: str
    monthly_token_budget: int | None = None
    admin_contact_email: str | None = None
