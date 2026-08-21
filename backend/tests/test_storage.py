import pytest

from magenta.storage import risk_model_path, tenant_model_dir, uplift_model_path


def test_paths_are_tenant_separated():
    assert tenant_model_dir("org_a") != tenant_model_dir("org_b")
    assert "org_a" in str(risk_model_path("org_a"))


def test_model_dir_honours_env_override(monkeypatch, tmp_path):
    """Railway mounts a volume outside the repo, so data_dir() is not reachable there."""
    monkeypatch.setenv("MAGENTA_MODEL_DIR", str(tmp_path))
    assert str(risk_model_path("org_a")).startswith(str(tmp_path))


def test_tenant_id_cannot_escape_the_model_root(monkeypatch, tmp_path):
    """A tenant id is an external identifier. It must never traverse."""
    monkeypatch.setenv("MAGENTA_MODEL_DIR", str(tmp_path))
    with pytest.raises(ValueError):
        risk_model_path("../../etc")
    with pytest.raises(ValueError):
        # On Windows, model_root() / "C:evil" resolves to WindowsPath("C:evil"),
        # silently discarding the model root -- a denylist misses this; only an
        # allowlist of safe characters catches it.
        risk_model_path("C:evil")


def test_risk_model_save_requires_an_explicit_path():
    from magenta.brain.risk import RiskModel

    with pytest.raises(TypeError):
        RiskModel().save()


def test_uplift_model_load_requires_an_explicit_path():
    from magenta.brain.uplift import UpliftModel

    with pytest.raises(TypeError):
        UpliftModel.load()
