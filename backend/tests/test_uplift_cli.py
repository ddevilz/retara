from typer.testing import CliRunner

from magenta.cli import app

runner = CliRunner()


def test_uplift_report_runs():
    r = runner.invoke(app, ["uplift", "report", "--n", "3000", "--seed", "31"])
    assert r.exit_code == 0, r.output
    assert "Qini" in r.output
    assert "PERSUADABLE" in r.output
    assert "decile" in r.output.lower()
