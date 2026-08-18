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


def test_bare_invocation_shows_help():
    result = runner.invoke(app, [])
    assert result.exit_code != 0 or "smoke" in result.output
    assert "Usage" in result.output or "smoke" in result.output


def test_no_sqlite_anywhere_in_src():
    """The migration is only done when the import is gone."""
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "magenta"
    offenders = [
        str(p.relative_to(src))
        for p in src.rglob("*.py")
        if "sqlite3" in p.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"sqlite3 still referenced in: {offenders}"
