"""The current tenant, as a context variable.

`llm.chat()` is called from graph nodes, the chat agent, and System-2, none of which
carry a tenant parameter. Adding one to every signature would be a sweeping diff through
code this phase does not otherwise touch. contextvars are task-local and copied into
threadpool workers by anyio, so both the async routes and the sync handlers see the right
value without any plumbing.
"""
from __future__ import annotations

from contextvars import ContextVar

import structlog

current_tenant_id: ContextVar[str | None] = ContextVar("current_tenant_id", default=None)


def set_tenant(tenant_id: str) -> None:
    """Set the tenant for this task, and bind it onto every subsequent log line."""
    current_tenant_id.set(tenant_id)
    structlog.contextvars.bind_contextvars(tenant_id=tenant_id)


def get_tenant() -> str | None:
    return current_tenant_id.get()


def require_tenant() -> str:
    tenant_id = current_tenant_id.get()
    if tenant_id is None:
        raise RuntimeError(
            "no tenant in context; an LLM call was made outside a tenant-scoped request or job"
        )
    return tenant_id
