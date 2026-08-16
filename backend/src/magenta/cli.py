"""magenta CLI (typer). Subcommands grow per lab. Entry point: magenta.cli:app."""

from __future__ import annotations

import sqlite3
import hashlib
import random
import time
from collections import Counter

import numpy as np
import typer
import uvicorn

from magenta.brain.bandit import ThompsonBandit
from magenta.brain.features import FEATURE_NAMES, featurize
from magenta.brain.parity import main as parity_main
from magenta.brain.policy import BrainPolicy
from magenta.brain.risk import RiskModel
from magenta.brain.training import build_training_data
from magenta.brain.uplift import Segment, UpliftModel, classify_segment
from magenta.chat.persona import Archetype, PersonaAgent, make_persona
from magenta.chat.runner import run_negotiation
from magenta.config import configs_dir, data_dir, load_models
from magenta.cost.cache import SemanticCache
from magenta.cost.cascade import cascade
from magenta.cost.meter import CostMeter
from magenta.db import get_conn
from magenta.evalx.golden import run_golden
from magenta.evalx.hardchecks import scan_guardrail_compliance, scan_holdout_purity
from magenta.evalx.judge import judge_sample
from magenta.experiment import Scorecard, run_experiment
from magenta.graph import batch_diagnose
from magenta.graph.ablation import RUNGS, make_policy, run_ladder, write_scorecards
from magenta.graph.build import GraphDeps, build_graph, open_sqlite_saver, persist_audit
from magenta.graph.nodes import _DIAGNOSE_SYSTEM, _diagnose_user_prompt, _observables
from magenta.graph.state import RiskUpliftReport, Timing
from magenta.graph.tables import DEFAULT_TENANT_ID
from magenta.llm import chat, chat_structured
from magenta.memory.embed import LocalEmbedder
from magenta.memory.eval import run_memory_eval
from magenta.memory.store import CustomerMemory
from magenta.offers import Arm, OfferCatalog, OfferDecision
from magenta.sim.oracle import ResponseOracle, SimParams
from magenta.sim.population import generate_population
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


@app.command("parity")
def parity() -> None:
    """Real-data parity check: IBM Telco CSV vs simulator, same pipeline.

    Downloads data/telco_real.csv once (skipped if present), trains the same
    LightGBM+isotonic pipeline as `magenta risk train` on it, and prints its
    AUC/Brier/ECE next to a fresh in-memory sim evaluation. Writes
    data/parity_report.txt. See backend/scripts/real_data_parity.py /
    magenta.brain.parity for the implementation.
    """
    parity_main()


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


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="bind host"),
    port: int = typer.Option(8000, help="bind port"),
    reload: bool = typer.Option(False, help="autoreload on source change (dev only)"),
) -> None:
    """Run the FastAPI + SSE server (local-first demo backend).

    Uses an import-string target ("magenta.api.app:app") rather than a direct
    module import, so `magenta.api.app` (and its FastAPI/CORS setup) is only
    constructed when `serve` actually runs and uvicorn's reloader can re-import
    it in a worker subprocess -- every other CLI command stays free of the
    FastAPI app-construction cost.
    """
    uvicorn.run(
        "magenta.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


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
    policy: str = typer.Option("rules", "--policy", help="|".join(RUNGS)),
    n: int = typer.Option(10000, "-n", "--n", help="population size"),
    seed: int = typer.Option(42, "--seed", help="seed (population + CRN + bootstrap)"),
    budget: float = typer.Option(None, "--budget", help="optional total offer-spend cap"),
) -> None:
    """Run a single-period two-arm RCT and print the Scorecard (ATE +/- bootstrap CI).

    `--policy` walks the ablation ladder (§7): noaction -> rules -> risk_rules
    -> agent_s1 -> agent. The two simple rungs need no ML/graph deps; the
    risk-gated and agent rungs build the same real GraphDeps `run-one` uses
    (loading persisted risk/uplift artifacts, training a stand-in if missing).
    """
    if policy not in RUNGS:
        typer.secho(f"unknown policy {policy!r}; choose from {RUNGS}",
                    fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    deps = None
    if policy in {"risk_rules", "agent_s1", "agent"}:
        conn = get_conn()
        _, hidden = generate_population(n=n, seed=seed)
        bandit = ThompsonBandit(dim=len(FEATURE_NAMES), arms=list(Arm), seed=seed)
        try:
            bandit.load(conn, DEFAULT_TENANT_ID)  # no-op prior if no rows yet
        except sqlite3.OperationalError:
            pass
        sim_params = SimParams.load(configs_dir() / "sim_params.yaml")
        deps = GraphDeps(
            risk=_load_or_train_risk(seed),
            uplift=_load_or_train_uplift(seed),
            bandit=bandit,
            catalog=OfferCatalog.load(configs_dir() / "offers.yaml"),
            oracle=ResponseOracle(hidden, params=sim_params, seed=seed),
            conn=conn, params=_GraphParams(), chat=_ChatShim(),
            load_customer=lambda cid: None,
        )
    sc = run_experiment(make_policy(policy, deps), n=n, seed=seed, budget=budget)
    typer.echo(_format_scorecard(sc, policy))


@app.command("ablation")
def ablation(
    n: int = typer.Option(10000, "-n", "--n", help="population size per rung"),
    seed: int = typer.Option(7, "--seed", help="seed (population + CRN + bootstrap)"),
) -> None:
    """Run the 5-rung ablation ladder (§7), print a comparison table, and
    write `data/scorecards.json` (the contract schema labs 10-11 read).

    Each rung gets a freshly-built `GraphDeps` (own conn/cold bandit prior)
    via the same real risk/uplift/catalog/oracle wiring `experiment
    --policy ...` uses; `noaction`/`rules` never touch it. `agent_s1`/`agent`
    call the real LLM (GROQ_API_KEY) through `_ChatShim` — mock
    `magenta.llm.chat`/`chat_structured`, or pass a stub via `GraphDeps.chat`
    directly (as the graph tests do), to run this offline.
    """
    def deps_factory(n_: int, seed_: int) -> GraphDeps:
        conn = get_conn()
        _, hidden = generate_population(n=n_, seed=seed_)
        bandit = ThompsonBandit(dim=len(FEATURE_NAMES), arms=list(Arm), seed=seed_)
        try:
            bandit.load(conn, DEFAULT_TENANT_ID)  # no-op prior if no rows yet
        except sqlite3.OperationalError:
            pass
        sim_params = SimParams.load(configs_dir() / "sim_params.yaml")
        return GraphDeps(
            risk=_load_or_train_risk(seed_),
            uplift=_load_or_train_uplift(seed_),
            bandit=bandit,
            catalog=OfferCatalog.load(configs_dir() / "offers.yaml"),
            oracle=ResponseOracle(hidden, params=sim_params, seed=seed_),
            conn=conn, params=_GraphParams(), chat=_ChatShim(),
            load_customer=lambda cid: None,
        )

    ladder = run_ladder(n=n, seed=seed, deps_factory=deps_factory)

    header = f"{'RUNG':<12}{'ATE':>10}{'CI':>22}{'WASTED':>9}{'SPEND':>12}{'€RETAINED':>13}"
    typer.echo(typer.style(header, bold=True))
    typer.echo("-" * len(header))
    for rung in RUNGS:
        s = ladder[rung]
        ci = f"[{s.ci_low:+.4f},{s.ci_high:+.4f}]"
        typer.echo(f"{rung:<12}{s.ate:>10.4f}{ci:>22}{s.wasted_offer_rate:>9.3f}"
                   f"{s.offer_spend:>12.1f}{s.euros_retained:>13.1f}")

    out_path = str(data_dir() / "scorecards.json")
    write_scorecards(out_path, ladder)
    typer.echo(typer.style(f"\nwrote {out_path}", fg="green"))
    # honesty guard (§10 risk #2): flag if agent fails to beat rules.
    if ladder["agent"].ate < ladder["rules"].ate:
        typer.echo(typer.style(
            "NOTE: agent did NOT beat rules on ATE this run — reporting honestly.",
            fg="yellow"))


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
    bandit = ThompsonBandit(dim=len(FEATURE_NAMES), arms=list(Arm), seed=seed)

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


## ---- appended by lab 6: single-customer graph walk (manual-test surface) ----


class _GraphParams:
    freq_cap_days = 14
    freq_cap_max = 1
    value_cap = 40.0
    p90_clv = 2000.0


class _ChatShim:
    """Adapts module-level llm.chat/chat_structured to the deps.chat interface."""

    def chat(self, role, messages, **kw):
        return chat(role, messages, **kw)

    def chat_structured(self, role, messages, model_cls):
        return chat_structured(role, messages, model_cls)


def _load_or_train_risk(seed: int) -> RiskModel:
    """RiskModel.load() with NO argument uses the data_dir()-anchored default
    path (fixes the brief's cwd-relative "data/risk.pkl" bug — that literal
    only resolves if the process cwd happens to be the repo root, but these
    commands are documented as `cd backend && uv run magenta ...`). If the
    artifact is missing, train a small stand-in on the fly so the manual-test
    surface still works end to end; `magenta risk train` remains the
    authoritative way to get a properly-sized model."""
    try:
        return RiskModel.load()
    except FileNotFoundError:
        typer.echo(
            "(no risk model artifact found; training a quick one on n=3000 -- "
            "run `magenta risk train` for a properly-sized model)"
        )
        td = build_training_data(n=3000, seed=seed)
        model = RiskModel().fit(td.customers, td.churned)
        model.save()
        return model


def _load_or_train_uplift(seed: int) -> UpliftModel:
    try:
        return UpliftModel.load()
    except FileNotFoundError:
        typer.echo(
            "(no uplift model artifact found; training a quick one on n=3000 -- "
            "no persisted `uplift train` command exists yet)"
        )
        td = build_training_data(n=3000, seed=seed)
        model = UpliftModel().fit(td.customers, td.treated, td.retained)
        model.save()
        return model


def _pretty(state: dict) -> None:
    def section(title, body):
        typer.echo(typer.style(f"\n=== {title} ===", fg="magenta", bold=True))
        typer.echo(body)

    r = state.get("risk")
    section("SENSE", f"p_churn={r.p_churn:.3f} band={r.band.value} "
                     f"segment={r.segment.value} tau={r.tau_hat:.3f} "
                     f"engage={r.engage} timing={r.timing.value}" if r else "(none)")
    if r and not r.engage:
        typer.echo(typer.style("\n— exited early: not engaged —", fg="yellow"))
        return
    d = state.get("diagnosis")
    if d:
        section("DIAGNOSE", f"tags={d.root_cause_tags} conf={d.confidence:.2f}\n{d.narrative}")
    o = state.get("offer")
    if o:
        section("DECIDE", f"arm={o.arm.value} cost={o.cost} propensity={o.propensity:.3f}")
    v = state.get("verdict")
    if v:
        section("GUARDRAIL", f"decision={v.decision} failed={v.failed_policies}")
    f = state.get("fulfillment")
    if f:
        status = f.get("STATUS") or f.get("status")
        if status == "SHADOW":
            section("ACT", "holdout shadow-log (no fulfillment)")
        else:
            section("ACT", f"status={status} key={f.get('IDEMPOTENCY_KEY') or f.get('idempotency_key')}")
    oc = state.get("outcome")
    if oc:
        section("OUTCOME", f"accepted={oc['accepted']} retained={oc['retained']} "
                           f"reward={oc['reward']:.2f}")


@app.command("run-one")
def run_one(customer_id: str,
            seed: int = typer.Option(7, help="population seed"),
            campaign: str = typer.Option("CAMP-A", help="campaign id"),
            holdout: bool = typer.Option(False, help="force holdout (shadow-log)")):
    """Step ONE customer through the graph and pretty-print every node."""
    customers, hidden = generate_population(n=1000, seed=seed)
    by_id = {c.customer_id: c for c in customers}
    if customer_id not in by_id:
        customer_id = customers[int(hashlib.sha256(customer_id.encode()).hexdigest(), 16) % len(customers)].customer_id
        typer.echo(f"(id not in seeded pop; using {customer_id})")
    customer = by_id.get(customer_id, customers[0])

    conn = get_conn()
    bandit = ThompsonBandit(dim=len(FEATURE_NAMES), arms=list(Arm), seed=seed)
    try:
        bandit.load(conn, DEFAULT_TENANT_ID)  # no-op prior if no rows yet
    except sqlite3.OperationalError:
        pass
    sim_params = SimParams.load(configs_dir() / "sim_params.yaml")
    deps = GraphDeps(
        risk=_load_or_train_risk(seed),
        uplift=_load_or_train_uplift(seed),
        bandit=bandit,
        catalog=OfferCatalog.load(configs_dir() / "offers.yaml"),
        oracle=ResponseOracle(hidden, params=sim_params, seed=seed),
        conn=conn, params=_GraphParams(), chat=_ChatShim(),
        load_customer=lambda cid: customer,
    )
    with open_sqlite_saver() as saver:
        deps.checkpointer = saver
        graph = build_graph(deps)
        init = {
            "customer_id": customer.customer_id, "campaign_id": campaign,
            "consent_flags": {"MARKETING": True},
            "risk": None, "diagnosis": None, "offer": None, "verdict": None,
            "fulfillment": None, "outcome": None, "messages": [], "audit_log": [],
            "requires_approval": False, "holdout": holdout,
        }
        final = graph.invoke(
            init, config={"configurable": {"thread_id": f"{customer.customer_id}:{campaign}"}})
    persist_audit(conn, deps.tenant_id, final.get("audit_log", []))
    deps.bandit.save(conn, deps.tenant_id)
    _pretty(final)


## ---- appended by lab 8 task 8.7: negotiation-runner chat subcommand ----

_ARCHETYPE_BY_FLAG = {
    "bill_shock": Archetype.BILL_SHOCK,
    "confused": Archetype.CONFUSED,
    "price_haggler": Archetype.PRICE_HAGGLER,
    "network_complainer": Archetype.NETWORK_COMPLAINER,
    "competitor_bluffer": Archetype.COMPETITOR_BLUFFER,
    "sleeping_dog": Archetype.SLEEPING_DOG,
}


@app.command("chat")
def chat_cmd(
    persona: str | None = typer.Option(None, help="One of: " + ", ".join(_ARCHETYPE_BY_FLAG)),
    seed: int = typer.Option(0, help="Population seed"),
    human: bool = typer.Option(False, "--human", help="Interactive stdin mode"),
    customer: str | None = typer.Option(None, help="Customer id for --human mode"),
) -> None:
    """Run one negotiation: a scripted persona (--persona ARCHETYPE) against
    the retention agent, or a live human (--human [--customer ID]) via stdin.
    """
    customers, hidden = generate_population(64, seed=seed)
    by_id = {c.customer_id: c for c in customers}
    target = by_id.get(customer, customers[0]) if human else customers[0]

    conn = get_conn()
    bandit = ThompsonBandit(dim=len(FEATURE_NAMES), arms=list(Arm), seed=seed)
    try:
        bandit.load(conn, DEFAULT_TENANT_ID)  # no-op prior if no rows yet
    except sqlite3.OperationalError:
        pass
    sim_params = SimParams.load(configs_dir() / "sim_params.yaml")
    deps = GraphDeps(
        risk=_load_or_train_risk(seed),
        uplift=_load_or_train_uplift(seed),
        bandit=bandit,
        catalog=OfferCatalog.load(configs_dir() / "offers.yaml"),
        oracle=ResponseOracle(hidden, params=sim_params, seed=seed),
        conn=conn, params=_GraphParams(), chat=_ChatShim(),
        load_customer=lambda cid: by_id.get(cid, target),
        campaign_id="CHAT",
    )

    if human:
        result = run_negotiation(deps, target, persona=None)
    else:
        arche = _ARCHETYPE_BY_FLAG.get(persona)
        if arche is None:
            raise typer.BadParameter(f"unknown persona '{persona}'")
        agent = PersonaAgent(make_persona(arche, target, hidden.get(target.customer_id)))
        result = run_negotiation(deps, target, persona=agent)

    typer.echo(f"\nstatus={result.status.value} turns={result.turns_used} "
               f"offer={result.offer_final.arm.value if result.offer_final else 'none'}")
    for t in result.transcript:
        typer.echo(f"  {t.speaker}: {t.text}")


## ---- appended by lab 9 task 9.4: `eval report` CLI (golden + hard checks) ----

eval_app = typer.Typer(help="Evaluation harness")
app.add_typer(eval_app, name="eval")


@eval_app.command("report")
def eval_report(
    judge: bool = typer.Option(False, "--judge", help="Also run a pairwise judge sample"),
) -> None:
    """Run golden scenarios + hard-check scans and print two tables.

    Exits 1 if ANY golden scenario fails or either hard-check scan reports a
    violation; exits 0 otherwise. `--judge` runs the LLM judge on a sample
    and prints a directional win-rate line but never affects the exit code
    (the judge is advisory, not a gate) -- passing an empty transcript
    sample (`k=0`) here is a placeholder wiring point until a real
    transcript corpus (lab 8 chat runs) is threaded through, so this branch
    makes no network call and needs no API key.
    """
    conn = get_conn()

    golden = run_golden()
    typer.echo("=== Golden scenarios ===")
    for r in golden:
        mark = "PASS" if r.passed else "FAIL"
        typer.echo(f"  [{mark}] {r.name}: {r.detail}")

    holdout_viol = scan_holdout_purity(conn)
    guardrail_viol = scan_guardrail_compliance(conn)
    typer.echo("\n=== Hard checks ===")
    typer.echo(f"  holdout_purity: {'PASS' if not holdout_viol else 'FAIL ' + str(holdout_viol)}")
    typer.echo(f"  guardrail_compliance: "
               f"{'PASS' if not guardrail_viol else 'FAIL ' + str(guardrail_viol)}")

    if judge:
        rep = judge_sample([], baseline_fn=lambda c: "", k=0)
        typer.echo(f"\n=== Judge (directional) ===\n  win_rate={rep.win_rate} ties={rep.ties}")

    golden_failed = any(not r.passed for r in golden)
    hard_failed = bool(holdout_viol) or bool(guardrail_viol)
    if golden_failed or hard_failed:
        raise typer.Exit(code=1)
    typer.echo("\nAll hard checks passed.")


## ---- appended by lab 12 tasks 12.5+12.6: temporal customer memory CLI ----


@app.command("memory")
def memory_cmd(action: str, customer_id: str = typer.Argument(None)):
    """show <customer_id> -- print a customer's ordered memory timeline.
    eval -- run the mini temporal-retrieval eval and print its accuracy.
    """
    if action == "show":
        if customer_id is None:
            typer.secho("usage: magenta memory show <customer_id>", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2)
        m = CustomerMemory(get_conn(), embedder=LocalEmbedder())
        m.init_tables()
        for e in m.timeline(customer_id):
            span = f"{e.valid_from}->{e.valid_to or 'now'}"
            typer.echo(f"[{span}] {e.subject} {e.relation} {e.object}")
    elif action == "eval":
        typer.echo(run_memory_eval())
    else:
        typer.secho(f"unknown memory action {action!r}; choose from: show, eval",
                    fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)


## ---- appended by lab 13 task 13.4: cost/rate cascade + cache CLI ----

cost_app = typer.Typer(help="LLM cost/rate cascade + cache reporting.")
app.add_typer(cost_app, name="cost")

_ROOT_CAUSE_TAGS = (
    "OVERAGE", "DROPPED_CALLS", "BILL_SHOCK", "COMPETITOR_OFFER",
    "CONTRACT_EXPIRY", "SERVICE_COMPLAINT",
)


def _extract_tag(text: str) -> str | None:
    """Best-effort root-cause tag extraction from a free-text diagnosis
    answer, for the quality-retained agreement check below (the cascade's
    chat_fn returns plain text, not a structured Diagnosis)."""
    upper = (text or "").upper()
    for tag in _ROOT_CAUSE_TAGS:
        if tag in upper:
            return tag
    return None


@cost_app.command("report")
def cost_report(
    n: int = typer.Option(200, "-n", "--n", help="population sample to draw the cohort from"),
    seed: int = typer.Option(7, help="population + risk/uplift seed"),
    sample: int = typer.Option(100, help="quality-retained sample size (cascade vs forced-large)"),
) -> None:
    """Run cohort diagnosis under the semantic cache + confidence cascade
    (Tasks 13.1-13.4), then print the CostMeter report plus a
    quality-retained score.

    Only PERSUADABLE/engaged customers are diagnosed, mirroring the graph's
    own engage-gate (`sense` -> `should_engage` -> `diagnose`).

    quality_retained = agreement rate, on a random sample, between the
    cascade's actual answer and forcing that same prompt straight to the
    large role -- an honest check that cache+cascade are not silently
    trading away diagnosis quality for cost.

    est_cost_per_decision ~= EUR0.00 (Groq free tier): the real budget is the
    <=30 RPM / 1k-req/day LARGE-role rate limit, not money -- the semantic
    cache and confidence cascade are the RATE levers that keep a cohort run
    under that cap, which is what this report actually measures.
    """
    customers, _ = generate_population(n, seed=seed)
    risk = _load_or_train_risk(seed)
    uplift = _load_or_train_uplift(seed)

    reports: dict[str, RiskUpliftReport] = {}
    for c in customers:
        a = risk.score(c)
        tau = uplift.tau(c)
        segment = classify_segment(a.p_churn, tau)
        if segment is not Segment.PERSUADABLE:
            continue  # mirrors the graph's engage-gate: only engaged customers reach diagnose
        timing = Timing.ACT_NOW if a.band.value in ("HIGH", "CRITICAL") else Timing.SNOOZE
        reports[c.customer_id] = RiskUpliftReport(
            p_churn=a.p_churn, band=a.band, drivers=a.drivers, tau_hat=tau,
            segment=segment, engage=True, timing=timing,
        )

    if not reports:
        typer.echo("no PERSUADABLE customers in this cohort -- nothing to diagnose.")
        raise typer.Exit(code=0)

    cache = SemanticCache(get_conn(), LocalEmbedder())
    meter = CostMeter()
    try:
        batch_diagnose.diagnose_cohort(customers, reports, deps=None, meter=meter, cache=cache)

        by_id = {c.customer_id: c for c in customers}
        rng = random.Random(seed)
        sample_ids = rng.sample(list(reports), k=min(sample, len(reports)))
        agree = 0
        for cid in sample_ids:
            messages = [
                {"role": "system", "content": _DIAGNOSE_SYSTEM},
                {"role": "user", "content": _diagnose_user_prompt(reports[cid], _observables(by_id[cid]))},
            ]
            cascade_answer = cascade(
                messages, batch_diagnose._chat, batch_diagnose._confidence_from_answer
            ).answer
            forced_large = batch_diagnose._chat("large", messages)
            if _extract_tag(cascade_answer) == _extract_tag(forced_large):
                agree += 1
        quality_retained = agree / len(sample_ids) if sample_ids else 1.0
    except Exception as exc:  # surface config/network errors cleanly, no stack trace
        typer.secho(
            f"cost report failed: {exc}\nHint: set GROQ_API_KEY (default provider) or OPENAI_API_KEY + optional OPENAI_BASE_URL.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    r = meter.report()
    typer.echo("=== Cost/rate report ===")
    typer.echo(f"  total_decisions      {r['total_decisions']}")
    typer.echo(f"  pct_routed_cheap     {r['pct_routed_cheap']:6.2%}")
    typer.echo(f"  cache_hit_rate       {r['cache_hit_rate']:6.2%}")
    typer.echo(f"  escalation_rate      {r['escalation_rate']:6.2%}")
    typer.echo(f"  quality_retained     {quality_retained:6.2%}  (n={len(sample_ids)} vs forced-large)")
    typer.echo(
        "  est_cost_per_decision  EUR0.00 (Groq free tier) -- the real budget "
        "is the <=30 RPM large-role rate limit; cache + cascade are the rate levers."
    )


if __name__ == "__main__":
    app()
