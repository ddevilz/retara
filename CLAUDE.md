# Magenta Retain — repo conventions (authoritative)

Telecom churn *retention* agent. Began as a DTDL AI Engineer hiring hackathon;
**since 2026-08-14 being built out as a real multi-tenant SaaS** — see
§Production direction (bottom), which governs all new work.
Thesis: predict whom we can **save** (uplift, not risk), pick the offer with a
bandit, negotiate via LLM, and **prove causation** against an untouched holdout.
Spec: `docs/superpowers/specs/2026-06-30-magenta-retain-design.md` (§0.5 governs).
**Note:** `docs/`, `.claude/`, `.agents/`, `.superpowers/` are **local-only,
gitignored by owner decision** — present on this machine, absent from a fresh
clone. Don't hunt for them in git history. Build ledger: `.superpowers/sdd/progress.md`.

## Layout (monorepo)
- `backend/` — uv project, Python 3.12, package `magenta` at `backend/src/magenta/`:
  - `sim/` population (hidden state in `HiddenStore`) · events · oracle (frozen params)
  - `brain/` features · risk (LightGBM+isotonic+SHAP) · uplift (S/T-learner) · bandit (Thompson) · policy · training · parity
  - `graph/` LangGraph decision graph: state · nodes (sense→diagnose→decide→guardrail→act→outcome) · build · policy (AgentPolicy) · system2 · ablation · batch_diagnose · tables · scenario
  - `chat/` negotiation: state · perceive · controller · ladder · agent (RetentionChat) · persona · runner
  - `evalx/` judge (pairwise, position-swap) · golden (11 pinned scenarios) · hardchecks
  - `api/` FastAPI+SSE: app · schemas · routes_{data,stream,chat} · deps · sse · chat_sessions
  - `memory/` temporal KG: embed (sentence-transformers, CPU) · store (MEMORY_EDGES) · eval
  - tests at `backend/tests/` (mirrors src; markers: `slow`, `hardcheck`)
- `frontend/` — Vite+React18+TS, Tailwind (magenta #E20074, dark), recharts;
  `src/api/` typed client+SSE; pages Overview/Customers/RunOne/Negotiation.
  Vite dev-proxies `/api` → `localhost:8000`.
- `configs/` — `models.yaml` (roles CHEAP/LARGE/JUDGE), `sim_params.yaml` (**FROZEN** — never tune), `offers.yaml` (8 arms).
- `data/` — `telco_marginals.json` (committed); everything else gitignored (db, models, scorecards.json, telco_real.csv, parity_report.txt).

## Hard conventions (do not violate)
- **Enum values ALL_CAPS** (`Arm.BILL_CREDIT`); **SQL table AND column names ALL_CAPS** —
  every table, every column, no exceptions. On Postgres this requires **double-quoting
  identifiers in DDL and in every query** (`"FULFILLMENTS"`, `"TENANT_ID"`): unquoted
  identifiers silently fold to lowercase, so `CREATE TABLE FULFILLMENTS` yields a table
  actually named `fulfillments` and later quoted references then fail to find it.
  Third-party schemas (`procrastinate_jobs`, `information_schema`) keep their own casing.
- **Pydantic v2** everywhere; **seeds everywhere** (same seed ⇒ identical output — explicit `seed`/`rng` params, sha256 not `hash()`).
- **Plain `openai` package** — NO LangChain/LiteLLM gateway. Groq default via `GROQ_API_KEY`; `OPENAI_API_KEY` wins if set. Per-role env override: `MAGENTA_MODEL_<CHEAP|LARGE|JUDGE>`.
- **LangSmith** via `langsmith.wrappers.wrap_openai` + `LANGSMITH_TRACING` (no LangChain).
- **No network in tests** — mock `magenta.llm.chat`/`chat_structured`. DB: real
  Postgres, including tests (see §Production direction) — the `:memory:` SQLite
  rule was retired in Phase 1.1.
- **All imports at module top** — NO function-level/lazy imports, ever (owner rule).
- **No module-level name shadowing** — a typer command `def chat(...)` once shadowed
  `magenta.llm.chat` and crashed a live cohort run. Command fns get `_cmd` suffixes
  when they'd collide; identity test pins `cli.chat is llm.chat`.
- **No per-customer TreeSHAP in cohort loops** — use `RiskModel.p_churn_batch` /
  `UpliftModel.tau_batch`; `.score()` (SHAP drivers) is for single customers.
- Paths anchor to `magenta.config.{repo_root,configs_dir,data_dir}()` — never cwd-relative.
- LLM calls must survive failure: `llm.py` has bounded 429 retry-after backoff;
  diagnose degrades to NO_ACTION path; System-2 degrades to S1 (`SYSTEM2_DEGRADED_S1` audit).

## Anti-circularity (the #1 critique defense)
Hidden state (`theta_churn_base`, `theta_price_sens`, `persuadable_segment`,
`competitor_pull`) lives ONLY in the simulator-private `HiddenStore`. NEVER a field
on `Customer`, never in graph state, never in a prompt, never a model feature.
Tests assert no leak (unit + end-to-end serialization sweep). `configs/sim_params.yaml`
is frozen; report the ablation ladder honestly (the CLI prints an honesty note when
the agent doesn't win). Parity check (`magenta parity`): same pipeline on real IBM
data AUC≈0.83 vs sim 0.72 — the sim is *harder* than reality, not easier.

## Console script (`magenta` = typer app in `magenta/cli.py`)
`smoke` · `sim generate` · `experiment --policy noaction|rules|risk_rules|agent_s1|agent`
· `risk train|eval|score` · `uplift report` · `bandit episodes` · `run-one <id>`
· `ablation` (5-rung ladder → data/scorecards.json) · `chat --persona X|--human`
· `eval report [--judge]` · `parity` · `memory show <id>|eval` · `serve` (FastAPI :8000).
Demo: `magenta serve` + `cd frontend && npm run dev` → localhost:5173.

## Working rules
- TDD (failing test → impl → pass → commit); review gate per task; fixes get regression tests.
- Frontend gates: `npm run typecheck && npm run test -- --run && npm run build`.
- Full backend suite is slow (LightGBM training) — prefer targeted `pytest tests/<area>/`;
  `-m "not slow"` skips the heavy ladder test.
- Never commit `.env` (real keys live there), model binaries, or `data/` artifacts.

## Production direction (decided 2026-08-14)
Pivot: hackathon demo → real multi-tenant SaaS. Driving milestone: **one design
partner using it on their own customer data.** Full spec (local-only, gitignored):
`docs/superpowers/specs/2026-08-14-production-platform-design.md`.

**Phase 1.1 is BUILT:** Postgres + SQLAlchemy Core + Alembic, `TENANT_ID` on
all six tables, tenant-scoped composite idempotency key, tests run on real
Postgres. **Phases 1.2 onward remain DESIGNED, NOT YET BUILT** — no Clerk
auth, no Procrastinate jobs, no per-tenant `get_graph_deps()`, no live mode.
Treat everything below this point as direction for those phases, not current
state, unless noted otherwise above.

Locked stack: Postgres + Alembic + SQLAlchemy Core (deliberate SQL retained, not
the ORM) · Clerk auth · **Procrastinate** for jobs (NOT Celery — transactional
enqueue, no Redis service) · Railway (web + worker + Postgres) · shared tables
with `TENANT_ID` · no billing until a buyer exists.

Invariants for the productized system:
- **Two modes, one graph** — an `OutcomeSource` seam. Sandbox = today's oracle
  (instant); live = real channel + delayed ingested outcome. `graph/nodes.py`
  currently calls `deps.oracle.outcome()` inline; live mode splits the graph at
  `act` and makes the bandit delayed-feedback.
- **Never build per-vertical simulators.** Generalize the *schema*, not the
  simulator — other verticals get a sandbox by replaying their own uploaded data.
- **Idempotency keys must include `TENANT_ID`** — DONE in Phase 1.1. The old
  `customer_id:campaign_id:arm` key was a global PK and collided across tenants,
  silently suppressing a real offer. `idempotency_key()` now hashes
  `tenant_id:customer_id:campaign_id:arm` and `FULFILLMENTS`' PK is composite
  `("TENANT_ID","IDEMPOTENCY_KEY")`. The same rule binds any NEW cross-tenant key:
  LangGraph `thread_id` is likewise `tenant_id:customer_id:campaign_id`.
- **No process singletons.** `get_graph_deps()` becomes per-tenant with a
  *bounded* cache — each entry holds LightGBM models plus a population.
- Anti-circularity survives the pivot: on real data the same hardchecks become
  train/serve leakage defense. `sim_params.yaml` stays frozen (sandbox only).

Phases: 1 platform foundations · 2 product shell (replaces the demo frontend) ·
3 real-data ingestion · 4 live mode · 5 multi-vertical (parked) · 6 billing (parked).

**Decided 2026-08-14** — tests move to **real Postgres** (docker-compose locally,
GH Actions service container), retiring the `:memory:` SQLite rule. Reason: two
SQL dialects would diverge exactly where Phase 1 is riskiest (tenant-scoped
composite idempotency key, `ON CONFLICT`, RLS). "No network in tests" is
unaffected — LLM calls stay mocked.
