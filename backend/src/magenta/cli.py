"""magenta CLI (typer). Subcommands grow per lab. Entry point: magenta.cli:app."""

from __future__ import annotations

import time

import typer

from magenta.config import load_models
from magenta.llm import chat

app = typer.Typer(help="Magenta Retain — churn retention agent CLI.")


@app.callback(invoke_without_command=True)
def callback() -> None:
    """Main app callback."""
    pass


@app.command()
def smoke() -> None:
    """One real chat round-trip on the CHEAP model; prints model, reply, latency."""
    model = load_models()["CHEAP"]
    messages = [{"role": "user", "content": "Reply with the single word: pong"}]
    start = time.perf_counter()
    try:
        reply = chat("cheap", messages, temperature=0.0)
    except Exception as exc:  # surface config/network errors cleanly, no stack trace
        typer.secho(
            f"smoke failed: {exc}\nHint: set GROQ_API_KEY (default provider) or OPENAI_API_KEY + optional OPENAI_BASE_URL.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    latency_ms = (time.perf_counter() - start) * 1000.0
    typer.echo(f"model:   {model}")
    typer.echo(f"reply:   {reply}")
    typer.echo(f"latency: {latency_ms:.0f} ms")


if __name__ == "__main__":
    app()
