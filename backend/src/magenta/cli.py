"""magenta CLI (typer). Subcommands grow per lab. Entry point: magenta.cli:app."""

from __future__ import annotations

import time
from collections import Counter

import numpy as np
import typer

from magenta.brain.bandit import ThompsonBandit
from magenta.brain.features import FEATURE_NAMES, featurize
from magenta.brain.policy import BrainPolicy
from magenta.brain.risk import RiskModel
from magenta.brain.training import build_training_data
from magenta.brain.uplift import UpliftModel, classify_segment
from magenta.config import configs_dir, data_dir, load_models
from magenta.experiment import (
    NoActionPolicy,
    RulesPolicy,
    Scorecard,
    run_experiment,
)
from magenta.llm import chat
from magenta.offers import Arm, OfferCatalog, OfferDecision
from magenta.sim.oracle import ResponseOracle, SimParams
from magenta.sim.population import Segment, generate_population
from magenta.sim.stats import format_stats, population_stats

app = typer.Typer(help="Magenta Retain — churn retention agent CLI.", no_args_is_help=True)


@app.callback()
def callback() -> None:
    """Magenta Retain — churn retention agent CLI."""


sim_app = typer.Typer(help="Simulator commands.", no_args_is_help=True)
app.add_typer(sim_app, name="sim")


risk_app = typer.Typer(help="Churn-risk model: train / eval / score.")
app.add_typer(risk_app, name="risk")


@risk_app.command("train")
def risk_train(
    n: int = typer.Option(8000, help="training population size"),
    seed: int = typer.Option(7, help="seed"),
    out: str = typer.Option(str(data_dir() / "models" / "risk.joblib"), help="output path"),
) -> None:
    """Train a churn-risk model on synthetic data."""
    td = build_training_data(n=n, seed=seed)
    model = RiskModel().fit(td.customers, td.churned)
    model.save(out)
    typer.echo(f"saved risk model -> {out} (n={n}, seed={seed})")


@risk_app.command("eval")
def risk_eval(
    n: int = typer.Option(4000, help="eval population size"),
    seed: int = typer.Option(99, help="seed"),
    model: str = typer.Option(str(data_dir() / "models" / "risk.joblib"), help="model path"),
) -> None:
    """Evaluate a risk model on a held-out population."""
    m = RiskModel.load(model)
    td = build_training_data(n=n, seed=seed)
    rep = m.evaluate(td.customers, td.churned)
    typer.echo("metric   value")
    typer.echo(f"AUC      {rep.auc:.4f}")
    typer.echo(f"Brier    {rep.brier:.4f}")
    typer.echo(f"ECE      {rep.ece:.4f}")


@risk_app.command("score")
def risk_score(
    customer_id: str = typer.Argument(..., help="customer id, e.g. SIM-0"),
    n: int = typer.Option(500, help="population to draw the customer from"),
    seed: int = typer.Option(7, help="seed"),
    model: str = typer.Option(str(data_dir() / "models" / "risk.joblib"), help="model path"),
) -> None:
    """Score a customer for churn risk."""
    m = RiskModel.load(model)
    customers, _ = generate_population(n, seed=seed)
    match = next((c for c in customers if c.customer_id == customer_id), None)
    if match is None:
        match = customers[0]
        typer.echo(f"(id {customer_id} not found; scoring {match.customer_id})")
    a = m.score(match)
    typer.echo(f"customer {match.customer_id}")
    typer.echo(f"p_churn  {a.p_churn:.4f}   band {a.band.value}")
    typer.echo("top drivers:")
    for d in a.drivers:
        arrow = "^" if d.direction == "UP" else "v"
        typer.echo(f"  {arrow} {d.label:<28} shap={d.shap_value:+.4f}")


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
        f"  EUROS RETAINED (NET)    {sc.euros_retained:10.2f}",
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


uplift_app = typer.Typer(help="Uplift model reporting.")
app.add_typer(uplift_app, name="uplift")


@uplift_app.command("report")
def uplift_report(
    n: int = typer.Option(6000, help="population size"),
    seed: int = typer.Option(31, help="seed"),
) -> None:
    """Train an uplift model and report segment counts, Qini score, and tau deciles."""
    td = build_training_data(n=n, seed=seed)
    um = UpliftModel().fit(td.customers, td.treated, td.retained)
    rm = RiskModel().fit(td.customers, td.churned)

    taus = um.tau_batch(td.customers)
    qini = um.qini(td.customers, td.treated, td.retained)

    counts: Counter = Counter()
    p_churns = rm.p_churn_batch(td.customers)
    for p, tau in zip(p_churns, taus):
        counts[classify_segment(float(p), float(tau)).value] += 1

    typer.echo(f"Qini (T-learner): {qini:.4f}")
    typer.echo("predicted segments:")
    for seg in ("PERSUADABLE", "SURE_THING", "LOST_CAUSE", "SLEEPING_DOG"):
        typer.echo(f"  {seg:<14} {counts.get(seg, 0)}")
    typer.echo("tau deciles:")
    deciles = np.percentile(taus, np.arange(0, 101, 10))
    for i, v in enumerate(deciles):
        typer.echo(f"  decile {i*10:>3}%  {v:+.4f}")


bandit_app = typer.Typer(help="Contextual bandit training loop.")
app.add_typer(bandit_app, name="bandit")


@bandit_app.command("episodes")
def bandit_episodes(
    episodes: int = typer.Option(30, "-e", "--episodes", help="number of episodes"),
    n: int = typer.Option(1000, help="cohort size per episode"),
    seed: int = typer.Option(3, help="base seed"),
) -> None:
    """Run E episodes (fresh cohort -> engage-gate + bandit arm choice -> oracle
    outcome -> bandit.update) with a SHARED, persistent bandit posterior and print
    the measured net-margin convergence curve.

    This is a measured-convergence demonstration over logged episodes, not a
    claim of live self-improvement in production.

    PERF: per-customer ``RiskModel.score()`` runs TreeSHAP and is ~100x slower
    than batch scoring, so each episode scores the whole cohort once via
    ``p_churn_batch``/``tau_batch`` and replicates BrainPolicy's engage-gate +
    arm-selection with the precomputed values, instead of calling
    ``BrainPolicy.decide()`` (which internally calls the slow per-customer
    ``score()``) in the hot loop.
    """
    # Train risk + uplift once on a held-out slice; the bandit learns online.
    td = build_training_data(n=max(n, 3000), seed=seed)
    rm = RiskModel().fit(td.customers, td.churned)
    um = UpliftModel().fit(td.customers, td.treated, td.retained)
    cat = OfferCatalog.load(configs_dir() / "offers.yaml")
    params = SimParams.load(configs_dir() / "sim_params.yaml")
    bandit = ThompsonBandit(dim=len(FEATURE_NAMES), arms=list(Arm))
    policy = BrainPolicy(rm, um, bandit, cat)

    typer.echo(
        "measured convergence across episodes (bandit posterior persists "
        "in-process; not a live self-improving claim)"
    )
    typer.echo(
        f"{'episode':<8} {'net_margin_per_intervention':<28} "
        f"{'NO_ACTION_share':<16} {'cumulative_net_margin':>22}"
    )
    cumulative_net_margin = 0.0
    for ep in range(1, episodes + 1):
        customers, hidden = generate_population(n, seed=seed + 100 + ep)
        oracle = ResponseOracle(hidden, params, seed=seed + 200 + ep)
        policy.reset_budget()

        # Batch score the whole cohort once (fast); no per-customer .score().
        p_churns = rm.p_churn_batch(customers)
        taus = um.tau_batch(customers)

        interventions = 0
        no_action = 0
        total_net = 0.0
        for c, p_churn, tau in zip(customers, p_churns, taus):
            # Mirrors BrainPolicy.decide()'s engage-gate: classify_segment's
            # own risk_floor (0.25) already subsumes BrainPolicy's separate
            # p_churn < _RISK_FLOOR check (same threshold), so PERSUADABLE
            # alone is the equivalent gate here.
            segment = classify_segment(float(p_churn), float(tau))
            if segment is not Segment.PERSUADABLE:
                no_action += 1
                continue

            eligible = [a for a in cat.eligible(c) if a != Arm.NO_ACTION]
            if not eligible:
                no_action += 1
                continue

            x = featurize(c)
            arm, propensity = bandit.select(x, eligible=eligible)
            decision = OfferDecision(
                arm=arm,
                cost=cat.cost(arm),
                rationale=f"persuadable p_churn={p_churn:.2f} tau={tau:.3f}",
                propensity=propensity,
            )
            out = oracle.outcome(c, decision)
            retained = 0 if out.churned else 1
            reward = retained * float(c.gross_margin_monthly) * 12 - decision.cost
            bandit.update(x, decision.arm, reward)
            total_net += reward
            interventions += 1

        npi = (total_net / interventions) if interventions else 0.0
        share = no_action / len(customers)
        cumulative_net_margin += total_net
        typer.echo(
            f"{ep:<8} {npi:<28.2f} {share:<16.3f} {cumulative_net_margin:>22.2f}"
        )


if __name__ == "__main__":
    app()
