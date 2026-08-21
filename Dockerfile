# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app/backend
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependencies first: this layer caches across source changes.
COPY backend/pyproject.toml backend/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY backend/ ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev


FROM python:3.12-slim AS runtime

# libgomp1 is required by LightGBM; the slim image does not ship it.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Layout must mirror the repo: config.py resolves repo_root() as parents[3] from
# /app/backend/src/magenta/config.py, so configs/ has to land at /app/configs.
WORKDIR /app
COPY --from=builder /app/backend /app/backend
COPY configs/ /app/configs/
COPY data/telco_marginals.json /app/data/telco_marginals.json

ENV PATH="/app/backend/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN useradd --create-home --uid 1000 magenta && chown -R magenta:magenta /app
USER magenta

EXPOSE 8000
CMD ["sh", "-c", "uvicorn magenta.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
