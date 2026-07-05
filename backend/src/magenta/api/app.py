"""FastAPI application factory."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

    # Routers land here incrementally as each is built:
    #   Task 10.2: from magenta.api.routes_data import router as data_router
    #              app.include_router(data_router)
    #   Task 10.3: from magenta.api.routes_stream import router as stream_router
    #              app.include_router(stream_router)
    #   Task 10.4: from magenta.api.routes_chat import router as chat_router
    #              app.include_router(chat_router)

    return app


app = create_app()
