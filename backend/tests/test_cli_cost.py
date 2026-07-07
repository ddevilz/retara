import magenta.cli as cli_mod
import magenta.cost.cascade as cascade_mod
import magenta.graph.batch_diagnose as batch_diagnose_mod
from magenta.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_help_lists_cost_report():
    result = runner.invoke(app, ["cost", "report", "--help"])
    assert result.exit_code == 0
    assert "quality" in result.output.lower() or "cost" in result.output.lower()


def test_cost_command_does_not_shadow_imports():
    """Regression (see test_chat_shim_binds_llm_chat_not_cli_command): `cost
    report` references the real batch_diagnose module + cascade function --
    neither is shadowed by a same-named CLI command/variable in magenta.cli."""
    assert cli_mod.batch_diagnose is batch_diagnose_mod, "batch_diagnose is shadowed in magenta.cli"
    assert cli_mod.cascade is cascade_mod.cascade, "cost.cascade.cascade is shadowed in magenta.cli"


def test_cost_report_runs_offline_with_mocked_chat(monkeypatch):
    """Full CLI wiring, no network: monkeypatch the single canonical chat_fn
    (magenta.graph.batch_diagnose._chat, the same injection point
    tests/cost/test_report.py uses) so `cost report` exercises the real
    cohort -> cache -> cascade -> meter -> quality-check path deterministically
    -- both diagnose_cohort's internal calls and cost_report's own
    quality-retained sampling loop resolve `_chat` through this one module
    attribute, so a single monkeypatch covers both. Risk/uplift artifacts are
    loaded from the real (gitignored) data/models/ -- same convention as the
    other GraphDeps-backed CLI commands (run-one, chat, experiment)."""
    monkeypatch.setattr(batch_diagnose_mod, "_chat", lambda role, msgs: "BILL_SHOCK overage credit")

    result = runner.invoke(app, ["cost", "report", "-n", "40", "--seed", "3", "--sample", "5"])
    assert result.exit_code == 0, result.output
    assert "total_decisions" in result.output
    assert "cache_hit_rate" in result.output
    assert "quality_retained" in result.output
    assert "est_cost_per_decision" in result.output
