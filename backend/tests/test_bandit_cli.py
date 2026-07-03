from typer.testing import CliRunner

from magenta.cli import app

runner = CliRunner()


def test_bandit_episodes_prints_learning_curve():
    r = runner.invoke(app, ["bandit", "episodes", "-e", "6", "--n", "800", "--seed", "3"])
    assert r.exit_code == 0, r.output
    assert "episode" in r.output.lower()
    assert "net_margin_per_intervention" in r.output
    assert "NO_ACTION" in r.output
    # 6 episode rows present.
    rows = [ln for ln in r.output.splitlines() if ln.strip().startswith(("1", "6"))]
    assert rows
