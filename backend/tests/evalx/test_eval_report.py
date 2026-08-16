"""Task 9.4: `magenta eval report` CLI -- golden + hard checks, exit codes.

SAFETY NOTE (deviation from the brief's literal test body): every test here
additionally patches `magenta.cli.get_conn` to hand back the shared `db_conn`
fixture connection (real Postgres, wrapped in a transaction that's rolled
back after the test). The brief's snippet calls the real (unmocked)
`get_conn()` inside `eval_report`; using the fixture connection instead of
letting `eval_report` open its own means every test stays inside the same
rolled-back transaction, so nothing it does here can leak into another test
or a concurrent `magenta ablation` run against the same DATABASE_URL.
`scan_holdout_purity` / `scan_guardrail_compliance` are mocked in every test
below anyway (the `conn` value they'd receive is never used for a real
query), but patching `get_conn` too means that stays true even if a future
edit here forgets to mock one of the scans -- `db_conn`'s tables are empty at
the start of every test, and the real scan functions vacuously return `[]`
for empty tables, so nothing would crash either.
"""
from unittest.mock import patch

from typer.testing import CliRunner

from magenta.cli import app
from magenta.evalx.golden import GoldenResult

runner = CliRunner()


def _all_pass():
    return [GoldenResult(name="a", passed=True, detail="ok"),
            GoldenResult(name="b", passed=True, detail="ok")]


def _one_fail():
    return [GoldenResult(name="a", passed=True, detail="ok"),
            GoldenResult(name="b", passed=False, detail="broke")]


def test_eval_report_exit_zero_when_all_pass(db_conn):
    with patch("magenta.cli.run_golden", return_value=_all_pass()), \
         patch("magenta.cli.scan_holdout_purity", return_value=[]), \
         patch("magenta.cli.scan_guardrail_compliance", return_value=[]), \
         patch("magenta.cli.get_conn", return_value=db_conn):
        res = runner.invoke(app, ["eval", "report"])
    assert res.exit_code == 0
    assert "PASS" in res.stdout or "passed" in res.stdout.lower()


def test_eval_report_exit_one_on_golden_fail(db_conn):
    with patch("magenta.cli.run_golden", return_value=_one_fail()), \
         patch("magenta.cli.scan_holdout_purity", return_value=[]), \
         patch("magenta.cli.scan_guardrail_compliance", return_value=[]), \
         patch("magenta.cli.get_conn", return_value=db_conn):
        res = runner.invoke(app, ["eval", "report"])
    assert res.exit_code == 1
    assert "broke" in res.stdout


def test_eval_report_exit_one_on_holdout_violation(db_conn):
    with patch("magenta.cli.run_golden", return_value=_all_pass()), \
         patch("magenta.cli.scan_holdout_purity", return_value=["CUST-9"]), \
         patch("magenta.cli.scan_guardrail_compliance", return_value=[]), \
         patch("magenta.cli.get_conn", return_value=db_conn):
        res = runner.invoke(app, ["eval", "report"])
    assert res.exit_code == 1
    assert "CUST-9" in res.stdout


def test_judge_flag_does_not_change_exit_code(db_conn):
    from magenta.evalx.judge import JudgeReport
    with patch("magenta.cli.run_golden", return_value=_all_pass()), \
         patch("magenta.cli.scan_holdout_purity", return_value=[]), \
         patch("magenta.cli.scan_guardrail_compliance", return_value=[]), \
         patch("magenta.cli.get_conn", return_value=db_conn), \
         patch("magenta.cli.judge_sample",
               return_value=JudgeReport(win_rate=0.4, ties=1, examples=[])):
        res = runner.invoke(app, ["eval", "report", "--judge"])
    assert res.exit_code == 0
    assert "0.4" in res.stdout or "win" in res.stdout.lower()
