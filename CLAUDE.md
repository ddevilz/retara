# Magenta Retain — repo conventions (authoritative)

Telecom churn *retention* agent (hiring hackathon → DTDL AI Engineer).
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
- **Enum values ALL_CAPS** (`Arm.BILL_CREDIT`); **SQLite table AND column names ALL_CAPS**.
- **Pydantic v2** everywhere; **seeds everywhere** (same seed ⇒ identical output — explicit `seed`/`rng` params, sha256 not `hash()`).
- **Plain `openai` package** — NO LangChain/LiteLLM gateway. Groq default via `GROQ_API_KEY`; `OPENAI_API_KEY` wins if set. Per-role env override: `MAGENTA_MODEL_<CHEAP|LARGE|JUDGE>`.
- **LangSmith** via `langsmith.wrappers.wrap_openai` + `LANGSMITH_TRACING` (no LangChain).
- **No network in tests** — mock `magenta.llm.chat`/`chat_structured`; `:memory:` SQLite.
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
