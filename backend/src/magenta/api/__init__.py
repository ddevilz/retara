"""HTTP transport layer for Magenta Retain (FastAPI + SSE).

Thin shell over the ``magenta`` package. No business logic lives here.
"""
from magenta.api.app import create_app

__all__ = ["create_app"]
