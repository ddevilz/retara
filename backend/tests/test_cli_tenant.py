from typer.testing import CliRunner

from magenta.cli import app
from magenta.storage import risk_model_path, uplift_model_path

runner = CliRunner()


def test_provision_writes_both_artifacts(monkeypatch, tmp_path):
    monkeypatch.setenv("MAGENTA_MODEL_DIR", str(tmp_path))
    result = runner.invoke(app, ["tenant", "provision", "org_new", "--n", "200"])
    assert result.exit_code == 0, result.output
    assert risk_model_path("org_new").exists()
    assert uplift_model_path("org_new").exists()


def test_provision_is_deterministic_per_tenant(monkeypatch, tmp_path):
    """Same tenant, same seed, same artifacts — the repo's determinism guarantee."""
    monkeypatch.setenv("MAGENTA_MODEL_DIR", str(tmp_path))
    runner.invoke(app, ["tenant", "provision", "org_det", "--n", "200"])
    first = risk_model_path("org_det").read_bytes()
    runner.invoke(app, ["tenant", "provision", "org_det", "--n", "200"])
    assert risk_model_path("org_det").read_bytes() == first
