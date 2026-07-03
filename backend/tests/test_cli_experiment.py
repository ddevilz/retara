from magenta.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_experiment_rules_prints_scorecard():
    result = runner.invoke(
        app, ["experiment", "--policy", "rules", "-n", "3000", "--seed", "42"])
    assert result.exit_code == 0
    out = result.output
    assert "SCORECARD" in out
    assert "ATE" in out
    assert "CI" in out
    assert "OFFERS MADE" in out


def test_experiment_noaction_zero_offers():
    result = runner.invoke(
        app, ["experiment", "--policy", "noaction", "-n", "3000", "--seed", "42"])
    assert result.exit_code == 0
    # Check that OFFERS MADE is rendered with value 0
    assert "OFFERS MADE" in result.output and "OFFERS MADE       0" in result.output


def test_experiment_bad_policy_errors():
    result = runner.invoke(
        app, ["experiment", "--policy", "wizard", "-n", "100", "--seed", "1"])
    assert result.exit_code != 0
