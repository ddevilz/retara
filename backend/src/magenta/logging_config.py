"""Structured JSON logging.

Context variables carry `request_id` and `tenant_id` onto every line without threading
them through call signatures. In a multi-tenant service, a log line without a tenant is
close to useless during an incident.
"""
from __future__ import annotations

import logging
import sys
import uuid
from typing import Any

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


def configure_logging() -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=logging.INFO)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    return structlog.get_logger(name)


def bind_tenant(tenant_id: str) -> None:
    structlog.contextvars.bind_contextvars(tenant_id=tenant_id)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Binds a request id for correlation and clears context afterwards.

    Clearing matters: contextvars are reused across requests in the same worker, so a
    leaked tenant_id would mislabel the next tenant's log lines.
    """

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        structlog.contextvars.clear_contextvars()
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            path=request.url.path,
            method=request.method,
        )
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()
        response.headers["X-Request-ID"] = request_id
        return response
