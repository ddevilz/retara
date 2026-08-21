"""FastAPI application factory."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from magenta.api.deps import ModelsNotReady
from magenta.api.routes_chat import router as chat_router
from magenta.api.routes_data import router as data_router
from magenta.api.routes_stream import router as stream_router
from magenta.api.schemas import Health
from magenta.db import get_conn
from magenta.jobs import app as procrastinate_app
from magenta.logging_config import RequestContextMiddleware, configure_logging, get_logger

logger = get_logger(__name__)


def allowed_origins() -> list[str]:
    """Comma-separated in the environment. Defaults to local dev so `magenta serve`
    keeps working with no configuration."""
    raw = os.environ.get("CORS_ALLOWED_ORIGINS", "")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    if "*" in origins:
        # CORSMiddleware is built below with allow_credentials=True. Starlette's
        # combination of a literal "*" origin with allow_credentials echoes the
        # request's actual Origin header back verbatim -- effectively
        # any-origin-with-credentials, a real footgun even though nothing sets
        # this today (render.yaml's CORS_ALLOWED_ORIGINS is unset). Fail loudly
        # at startup rather than silently accepting a config mistake that's
        # trivial to fix (name a real origin, or a comma-separated list).
        raise RuntimeError(
            'CORS_ALLOWED_ORIGINS="*" is not allowed with allow_credentials=True '
            "(effectively any-origin-with-credentials). Set it to a comma-separated "
            "list of real origins instead."
        )
    return origins or ["http://localhost:5173", "http://127.0.0.1:5173"]


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # `procrastinate_app.open()`/`.close()` are sync methods (even on this async
    # PsycopgConnector app -- see magenta/jobs.py) and must run once before any
    # `.defer()` call, including the sync ones `ensure_org` makes through a
    # caller-supplied connection.
    procrastinate_app.open()
    try:
        yield
    finally:
        procrastinate_app.close()


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="Magenta Retain API", version="0.1.0", lifespan=_lifespan)

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health", response_model=Health)
    def health() -> Health:
        """Liveness only. Must not touch the database."""
        return Health()

    @app.get("/api/ready")
    def ready():
        """Readiness: is the database reachable and migrated?

        Per-tenant model state is deliberately not checked -- models are per-tenant
        since Phase 1.3, so there is no global answer. An unprovisioned tenant already
        gets a 503 from ModelsNotReady.
        """
        try:
            with get_conn() as conn:
                conn.execute(text('SELECT 1 FROM "ORGANIZATIONS" LIMIT 1'))
        except Exception as exc:
            logger.error("readiness.failed", error=str(exc))
            return JSONResponse(status_code=503, content={"status": "not-ready"})
        return {"status": "ready"}

    @app.exception_handler(ModelsNotReady)
    async def models_not_ready_handler(request: Request, exc: ModelsNotReady):
        return JSONResponse(
            status_code=503,
            content={"detail": "models are still being prepared for this workspace"},
            headers={"Retry-After": "30"},
        )

    app.include_router(data_router)
    app.include_router(stream_router)
    app.include_router(chat_router)

    return app


app = create_app()
