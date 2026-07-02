"""Config loading + repo-root discovery. Configs live at <repo_root>/configs/."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Repo root = parent of the backend/ dir that holds this package."""
    # this file: <root>/backend/src/magenta/config.py
    return Path(__file__).resolve().parents[3]


def configs_dir() -> Path:
    return repo_root() / "configs"


def data_dir() -> Path:
    return repo_root() / "data"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"expected a mapping in {path}, got {type(data).__name__}")
    return data


def load_models() -> dict[str, str]:
    """Role -> model id. Keys CHEAP/LARGE/JUDGE."""
    models = load_yaml(configs_dir() / "models.yaml")
    required = {"CHEAP", "LARGE", "JUDGE"}
    missing = required - set(models)
    if missing:
        raise ValueError(f"models.yaml missing roles: {sorted(missing)}")
    return {k: str(models[k]) for k in required}
