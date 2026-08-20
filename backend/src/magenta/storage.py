"""Per-tenant model artifact locations.

MAGENTA_MODEL_DIR exists because Railway mounts a persistent volume at a path outside
the repo, where `data_dir()` (which is `repo_root()/data`) does not reach.
"""
from __future__ import annotations

import os
from pathlib import Path

from magenta.config import data_dir


def model_root() -> Path:
    override = os.environ.get("MAGENTA_MODEL_DIR")
    return Path(override) if override else data_dir() / "tenants"


def tenant_model_dir(tenant_id: str) -> Path:
    """Reject anything that is not a bare identifier. `tenant_id` arrives from a Clerk
    token; treating it as a path component without validation is a traversal."""
    if not tenant_id or "/" in tenant_id or "\\" in tenant_id or tenant_id.startswith("."):
        raise ValueError(f"unsafe tenant_id for a path component: {tenant_id!r}")
    path = model_root() / tenant_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def risk_model_path(tenant_id: str) -> Path:
    return tenant_model_dir(tenant_id) / "risk.joblib"


def uplift_model_path(tenant_id: str) -> Path:
    return tenant_model_dir(tenant_id) / "uplift.joblib"
