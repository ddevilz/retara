from pathlib import Path

from magenta.config import load_models, repo_root


def test_repo_root_contains_configs_dir():
    root = repo_root()
    assert (root / "configs" / "models.yaml").is_file()


def test_load_models_has_all_roles():
    models = load_models()
    assert set(models) == {"CHEAP", "LARGE", "JUDGE"}
    assert models["CHEAP"] == "llama-3.1-8b-instant"
    assert models["LARGE"] == "llama-3.3-70b-versatile"
    assert models["JUDGE"] == "openai/gpt-oss-120b"
    assert all(isinstance(v, str) and v for v in models.values())
