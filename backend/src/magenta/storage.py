"""Per-tenant model artifact locations.

MAGENTA_MODEL_DIR exists because Railway mounts a persistent volume at a path outside
the repo, where `data_dir()` (which is `repo_root()/data`) does not reach.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from magenta.config import data_dir


def model_root() -> Path:
    override = os.environ.get("MAGENTA_MODEL_DIR")
    return Path(override) if override else data_dir() / "tenants"


_SAFE_TENANT_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def tenant_model_dir(tenant_id: str) -> Path:
    """Allow only a bare identifier. `tenant_id` arrives from a Clerk token; treating
    it as a path component without validation is a traversal. An allowlist, not a
    denylist: a denylist of "/", "\\", leading "." missed "~", embedded NUL, and (on
    Windows) drive-relative paths like "C:evil", which `model_root() / "C:evil"`
    resolves to `WindowsPath("C:evil")`, silently discarding the model root.

    Pure path computation -- does not create the directory. `RiskModel.save` /
    `UpliftModel.save` create their own parent dir; a read path (cold ModelsNotReady)
    must not leave an empty directory behind for a tenant with no artifacts.
    """
    if not _SAFE_TENANT_ID.fullmatch(tenant_id):
        raise ValueError(f"unsafe tenant_id for a path component: {tenant_id!r}")
    return model_root() / tenant_id


def risk_model_path(tenant_id: str) -> Path:
    return tenant_model_dir(tenant_id) / "risk.joblib"


def uplift_model_path(tenant_id: str) -> Path:
    return tenant_model_dir(tenant_id) / "uplift.joblib"
