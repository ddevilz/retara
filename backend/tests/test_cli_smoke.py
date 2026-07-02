import magenta.cli as cli_mod
from magenta.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_help_lists_smoke():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "smoke" in result.output


def test_smoke_prints_model_reply_latency(monkeypatch):
    monkeypatch.setattr(cli_mod, "chat", lambda role, messages, **kw: "pong")
    result = runner.invoke(app, ["smoke"])
    assert result.exit_code == 0
    assert "llama-3.1-8b-instant" in result.output  # CHEAP model id
    assert "pong" in result.output
    assert "ms" in result.output
