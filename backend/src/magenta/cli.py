"""magenta CLI (typer). Subcommands grow per lab. Entry point: magenta.cli:app."""

from __future__ import annotations

import time

import typer

from magenta.config import configs_dir, load_models
from magenta.llm import chat
from magenta.sim.oracle import ResponseOracle, SimParams
from magenta.sim.population import generate_population
from magenta.sim.stats import format_stats, population_stats

app = typer.Typer(help="Magenta Retain — churn retention agent CLI.", no_args_is_help=True)


@app.callback()
def callback() -> None:
    """Magenta Retain — churn retention agent CLI."""


sim_app = typer.Typer(help="Simulator commands.", no_args_is_help=True)
app.add_typer(sim_app, name="sim")


@sim_app.command("generate")
def sim_generate(
    n: int = typer.Option(10000, "-n", "--n", help="number of customers"),
    seed: int = typer.Option(42, "--seed", help="population + oracle seed"),
    stats: bool = typer.Option(False, "--stats", help="print summary stats"),
) -> None:
    """Generate a synthetic population; optionally print distribution stats."""
    customers, hidden = generate_population(n, seed=seed)
    if stats:
        params = SimParams.load(configs_dir() / "sim_params.yaml")
        oracle = ResponseOracle(hidden, params, seed=seed)
        typer.echo(format_stats(population_stats(customers, hidden, oracle)))
    else:
        typer.echo(f"generated {len(customers)} customers (seed={seed})")


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
