# Magenta Retain — repo conventions (authoritative)

Telecom churn *retention* agent (hiring hackathon → DTDL AI Engineer).
Spec: `docs/superpowers/specs/2026-06-30-magenta-retain-design.md` (§0.5 Build contract governs).
Plan fragments: `docs/superpowers/plans/fragments/`.
**Note:** `docs/` (and `.claude/`, `.agents/`, `.superpowers/`) are **local-only, gitignored by owner decision** — they exist on this machine but not in a fresh clone. Don't hunt for them in git history.

## Layout (monorepo)
- `backend/` — uv project, Python 3.12. Package `magenta` at `backend/src/magenta/`, tests at `backend/tests/`.
- `frontend/` — React + Vite + TS (later labs).
- `configs/` — `models.yaml`, `sim_params.yaml`, `offers.yaml` (committed; frozen params live here).
- `data/` — `telco_marginals.json` (committed); `magenta.db` (gitignored).
- `docs/` — spec + plans.

## Hard conventions (do not violate)
- **Enum values are ALL_CAPS** (`Arm.BILL_CREDIT = "BILL_CREDIT"`).
- **SQLite table AND column names are ALL_CAPS** (`CUSTOMERS`, `CUSTOMER_ID`).
- **Pydantic v2** everywhere for schemas.
- **Seeds everywhere** — every stochastic function takes an explicit `seed`/`rng`; same seed ⇒ identical output.
- **Plain `openai` package** — NO LangChain / LiteLLM gateway. `base_url` keeps it swappable.
- **LangSmith** via `langsmith.wrappers.wrap_openai` + env `LANGSMITH_TRACING` (no LangChain).
- **No network in tests** — mock the OpenAI client.
- **All imports at module top** — NO function-level / lazy imports, ever (owner rule).

## Anti-circularity (the #1 critique defense)
Hidden state (`theta_churn_base`, `theta_price_sens`, `persuadable_segment`, `competitor_pull`)
lives ONLY in the simulator-private `HiddenStore`. It is NEVER a field on `Customer`, never
serialized into any graph/agent state, never in a prompt. A test asserts no leak.

## Console script
`magenta` = typer app in `magenta/cli.py`. Subcommands grow per lab: `smoke`, `sim`, `experiment`, …
