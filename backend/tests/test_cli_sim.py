from typer.testing import CliRunner

from magenta.cli import app

runner = CliRunner()


def test_sim_generate_stats_prints_summary():
    result = runner.invoke(app, ["sim", "generate", "-n", "800", "--seed", "42", "--stats"])
    assert result.exit_code == 0
    out = result.output
    assert "SEGMENT MIX" in out
    assert "PERSUADABLE" in out
    assert "CHURN BASE RATE" in out
    assert "MONTH_TO_MONTH" in out


def test_sim_generate_stats_reproducible():
    a = runner.invoke(app, ["sim", "generate", "-n", "800", "--seed", "42", "--stats"])
    b = runner.invoke(app, ["sim", "generate", "-n", "800", "--seed", "42", "--stats"])
    assert a.output == b.output
