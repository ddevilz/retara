"""FastAPI application factory."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from magenta.api.deps import ModelsNotReady
from magenta.api.routes_chat import router as chat_router
from magenta.api.routes_data import router as data_router
from magenta.api.routes_stream import router as stream_router
from magenta.api.schemas import Health

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def create_app() -> FastAPI:
    app = FastAPI(title="Magenta Retain API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health", response_model=Health)
    def health() -> Health:
        return Health()

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
