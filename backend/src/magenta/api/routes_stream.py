"""Streaming endpoints: live single-customer pipeline + experiment progress.

Two brief-bug fixes applied on sight, beyond the ones called out in the task:

1. `graph.stream(...)` needs a `config={"configurable": {"thread_id": ...}}`
   when the compiled graph has a checkpointer (it always does — `build_graph`
   falls back to an `InMemorySaver` when `deps.checkpointer` is None). Every
   other call site in the repo (cli.py, graph/policy.py, graph/scenario.py)
   passes this; omitting it raises `ValueError: Checkpointer requires one or
   more of the following 'configurable' keys: thread_id, ...` (confirmed
   against the installed langgraph version). `_CAMPAIGN_ID` gives the API's
   runs their own namespace, distinct from the CLI's "CAMP-A"/"AGENT-EXP".
2. `make_policy(req.policy, deps)` converts the request's policy string into
   a real `Policy` object before calling `run_experiment` — the brief passed
   `req.policy` (a plain string) straight to `run_experiment`, which calls
   `policy.decide(c)` and would crash on every real (non-mocked) call.
   Mirrors `magenta.cli.experiment`: only risk_rules/agent_s1/agent need a
   real GraphDeps; noaction/rules never touch it, so we don't pay for model
   loading on the cheap rungs.
"""
from __future__ import annotations

import anyio
from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from magenta.api.deps import find_customer, get_graph_deps
from magenta.api.schemas import ExperimentRequest, RunOneRequest
from magenta.api.sse import sse_event
from magenta.auth import TenantContext, current_tenant
from magenta.experiment import run_experiment
from magenta.graph.ablation import make_policy
from magenta.graph.build import persist_audit, build_graph

router = APIRouter(prefix="/api", tags=["stream"])

# Namespaces this endpoint's checkpointer threads / audit rows, distinct from
# the CLI's "CAMP-A" (run-one) / "AGENT-EXP" (GraphDeps default) campaigns.
_CAMPAIGN_ID = "API-RUN-ONE"

# Ladder rungs that need a real GraphDeps (risk model, catalog, oracle, LLM
# shim); the two simple rungs never touch deps (mirrors magenta.cli.experiment).
_DEPS_REQUIRED_POLICIES = {"risk_rules", "agent_s1", "agent"}


def _find_customer(customer_id: str):
    return find_customer(customer_id)


@router.post("/run-one")
async def run_one(
    req: RunOneRequest,
    tenant: TenantContext = Depends(current_tenant),
):
    tenant_id = tenant.tenant_id  # capture before the closure
    customer = _find_customer(req.customer_id)

    async def gen():
        if customer is None:
            yield sse_event("error", {"message": f"unknown customer {req.customer_id}"})
            yield sse_event("done", {"customer_id": req.customer_id})
            return

        deps = get_graph_deps()
        graph = build_graph(deps)
        # The graph is sync (langgraph .stream). Run it off the event loop so we
        # don't block; forward each node update as an SSE 'node' event.
        initial = {
            "customer_id": customer.customer_id,
            "campaign_id": _CAMPAIGN_ID,
            "consent_flags": {"MARKETING": True},
            "risk": None,
            "diagnosis": None,
            "offer": None,
            "verdict": None,
            "fulfillment": None,
            "outcome": None,
            "messages": [],
            "audit_log": [],
            "requires_approval": False,
            "holdout": False,
        }
        config = {"configurable": {"thread_id": f"{tenant_id}:{customer.customer_id}:{_CAMPAIGN_ID}"}}

        def _iter_updates(sink):
            for chunk in graph.stream(initial, config=config, stream_mode=["updates"]):
                # stream_mode=["updates"] yields (mode, update) tuples.
                mode, update = chunk if isinstance(chunk, tuple) else ("updates", chunk)
                for node, payload in update.items():
                    sink.append((node, payload))

        # Collect synchronously in a worker thread, then emit. Keeps ordering
        # simple and avoids cross-thread async queues for the demo scale.
        sink: list[tuple[str, object]] = []
        await anyio.to_thread.run_sync(_iter_updates, sink)
        audit_rows: list[dict] = []
        for node, payload in sink:
            if isinstance(payload, dict):
                audit_rows.extend(payload.get("audit_log") or [])
            yield sse_event("node", {"node": node, "payload": payload})
            await anyio.sleep(0)  # cooperative yield so client sees streaming
        # Persist the audit trail so the 10.2 customer-360 view shows this run.
        conn = getattr(deps, "conn", None)
        if conn is not None and audit_rows:
            persist_audit(conn, tenant_id, audit_rows)
        yield sse_event("done", {"customer_id": req.customer_id})

    return EventSourceResponse(gen())


@router.post("/experiment")
async def experiment(
    req: ExperimentRequest,
    tenant: TenantContext = Depends(current_tenant),
):
    async def gen():
        yield sse_event("progress", {"phase": "start", "policy": req.policy,
                                      "n": req.n, "seed": req.seed})

        def _run():
            deps = get_graph_deps() if req.policy in _DEPS_REQUIRED_POLICIES else None
            policy = make_policy(req.policy, deps)
            return run_experiment(policy, req.n, req.seed)

        yield sse_event("progress", {"phase": "running"})
        scorecard = await anyio.to_thread.run_sync(_run)
        yield sse_event("progress", {"phase": "scoring"})
        # scorecard is a pydantic Scorecard — sse.to_json handles model_dump.
        yield sse_event("scorecard", scorecard)
        yield sse_event("done", {"policy": req.policy})

    return EventSourceResponse(gen())
