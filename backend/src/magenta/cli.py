"""magenta CLI (typer). Subcommands grow per lab. Entry point: magenta.cli:app."""

from __future__ import annotations

import time

import typer

from magenta.config import configs_dir, load_models
from magenta.experiment import (
    NoActionPolicy,
    RulesPolicy,
    Scorecard,
    run_experiment,
)
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


def _format_scorecard(sc: Scorecard, policy: str) -> str:
    lines = [
        f"SCORECARD (policy={policy})",
        "",
        f"  N TREATMENT       {sc.n_treatment}",
        f"  N HOLDOUT         {sc.n_holdout}",
        f"  CHURN TREATMENT   {sc.churn_treatment:6.2%}",
        f"  CHURN HOLDOUT     {sc.churn_holdout:6.2%}",
        f"  ATE               {sc.ate:+.4f}",
        f"  CI [95%]          [{sc.ci_low:+.4f}, {sc.ci_high:+.4f}]",
        f"  OFFERS MADE       {sc.offers_made}",
        f"  ACCEPTANCE RATE   {sc.acceptance_rate:6.2%}",
        f"  WASTED OFFER RATE {sc.wasted_offer_rate:6.2%}",
        f"  SLEEPING DOGS HIT {sc.sleeping_dogs_contacted}",
        f"  OFFER SPEND       {sc.offer_spend:10.2f}",
        f"  EUROS RETAINED    {sc.euros_retained:10.2f}",
    ]
    return "\n".join(lines)


@app.command()
def experiment(
    policy: str = typer.Option("rules", "--policy", help="rules | noaction"),
    n: int = typer.Option(10000, "-n", "--n", help="population size"),
    seed: int = typer.Option(42, "--seed", help="seed (population + CRN + bootstrap)"),
    budget: float = typer.Option(None, "--budget", help="optional total offer-spend cap"),
) -> None:
    """Run a single-period two-arm RCT and print the Scorecard (ATE +/- bootstrap CI)."""
    policies = {"rules": RulesPolicy, "noaction": NoActionPolicy}
    if policy not in policies:
        typer.secho(f"unknown policy {policy!r}; choose from {sorted(policies)}",
                    fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    sc = run_experiment(policies[policy](), n=n, seed=seed, budget=budget)
    typer.echo(_format_scorecard(sc, policy))


if __name__ == "__main__":
    app()
