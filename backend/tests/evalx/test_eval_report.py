"""Task 9.4: `magenta eval report` CLI -- golden + hard checks, exit codes.

SAFETY NOTE (deviation from the brief's literal test body): every test here
additionally patches `magenta.cli.get_conn` to hand back a fresh `:memory:`
sqlite connection. The brief's snippet calls the real (unmocked) `get_conn()`
inside `eval_report`, which by default opens `<repo_root>/data/magenta.db` --
the app's shared, gitignored DB. A long-running live ablation
(`magenta ablation`) can be writing to that exact file while this suite runs,
so these tests must never open it, even read-only. `scan_holdout_purity` /
`scan_guardrail_compliance` are mocked in every test below anyway (the
`conn` value they'd receive is never used for a real query), but patching
`get_conn` too means that stays true even if a future edit here forgets to
mock one of the scans -- the in-memory conn has no tables, and the real scan
functions vacuously return `[]` for a missing-table conn (see
`magenta.evalx.hardchecks._table_exists`), so nothing would crash either.
"""
import sqlite3
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


def test_eval_report_exit_zero_when_all_pass():
    with patch("magenta.cli.run_golden", return_value=_all_pass()), \
         patch("magenta.cli.scan_holdout_purity", return_value=[]), \
         patch("magenta.cli.scan_guardrail_compliance", return_value=[]), \
         patch("magenta.cli.get_conn", return_value=sqlite3.connect(":memory:")):
        res = runner.invoke(app, ["eval", "report"])
    assert res.exit_code == 0
    assert "PASS" in res.stdout or "passed" in res.stdout.lower()


def test_eval_report_exit_one_on_golden_fail():
    with patch("magenta.cli.run_golden", return_value=_one_fail()), \
         patch("magenta.cli.scan_holdout_purity", return_value=[]), \
         patch("magenta.cli.scan_guardrail_compliance", return_value=[]), \
         patch("magenta.cli.get_conn", return_value=sqlite3.connect(":memory:")):
        res = runner.invoke(app, ["eval", "report"])
    assert res.exit_code == 1
    assert "broke" in res.stdout


def test_eval_report_exit_one_on_holdout_violation():
    with patch("magenta.cli.run_golden", return_value=_all_pass()), \
         patch("magenta.cli.scan_holdout_purity", return_value=["CUST-9"]), \
         patch("magenta.cli.scan_guardrail_compliance", return_value=[]), \
         patch("magenta.cli.get_conn", return_value=sqlite3.connect(":memory:")):
        res = runner.invoke(app, ["eval", "report"])
    assert res.exit_code == 1
    assert "CUST-9" in res.stdout


def test_judge_flag_does_not_change_exit_code():
    from magenta.evalx.judge import JudgeReport
    with patch("magenta.cli.run_golden", return_value=_all_pass()), \
         patch("magenta.cli.scan_holdout_purity", return_value=[]), \
         patch("magenta.cli.scan_guardrail_compliance", return_value=[]), \
         patch("magenta.cli.get_conn", return_value=sqlite3.connect(":memory:")), \
         patch("magenta.cli.judge_sample",
               return_value=JudgeReport(win_rate=0.4, ties=1, examples=[])):
        res = runner.invoke(app, ["eval", "report", "--judge"])
    assert res.exit_code == 0
    assert "0.4" in res.stdout or "win" in res.stdout.lower()
