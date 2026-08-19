"""In-memory chat session registry.

Non-persistent by design (spec: "sessions in-memory dict"). A session wraps
one RetentionChat instance so per-turn state (dialogue state, ladder position,
authority cap) persists across turn requests within a process lifetime.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChatSession:
    session_id: str
    tenant_id: str
    mode: str                 # "persona" | "human"
    customer_id: str
    archetype: Optional[str]
    chat: object              # magenta.chat.agent.RetentionChat
    persona: object = None    # magenta.chat.persona.PersonaBrief | None
    history: list = field(default_factory=list)  # [{"role","text"}]


_SESSIONS: dict[str, ChatSession] = {}


def create(session: ChatSession) -> None:
    _SESSIONS[session.session_id] = session


def get(session_id: str, tenant_id: str) -> Optional[ChatSession]:
    """Tenant mismatch is indistinguishable from "not found" by design — a caller
    must not be able to probe for the existence of another tenant's sessions."""
    session = _SESSIONS.get(session_id)
    if session is None or session.tenant_id != tenant_id:
        return None
    return session


def new_id() -> str:
    return f"sess-{uuid.uuid4().hex[:12]}"


def clear() -> None:  # test hook
    _SESSIONS.clear()
