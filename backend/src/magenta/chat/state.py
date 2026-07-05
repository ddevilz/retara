from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from magenta.offers import Arm


class ChatStatus(str, Enum):
    ACTIVE = "ACTIVE"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    ESCALATED = "ESCALATED"
    HANDOFF = "HANDOFF"


class Turn(BaseModel):
    speaker: str
    text: str


class DialogueState(BaseModel):
    customer_id: str
    turns: list[Turn] = Field(default_factory=list)
    intent_stack: list[str] = Field(default_factory=list)
    sentiment: float = 0.0
    open_slots: dict = Field(default_factory=dict)
    commitments: list[str] = Field(default_factory=list)
    understanding_confidence: float = 0.0
    ladder_position: int = 0
    authority_cap: float | None = None
    concessions_made: list[Arm] = Field(default_factory=list)
    status: ChatStatus = ChatStatus.ACTIVE
