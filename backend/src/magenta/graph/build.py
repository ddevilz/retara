"""Assemble the retention decision graph (§5.5).

Topology (explicit StateGraph — NOT supervisor/swarm; compliance wants a legible
state machine):

    START -> sense
    sense --(engage?)--> diagnose | END        # engage-gate = cost firewall
    diagnose -> decide -> guardrail
    guardrail --(pass/needs_approval?)--> act | END   # REJECT stops here
    act -> outcome -> END

thread_id = f"{customer_id}:{campaign_id}". SqliteSaver checkpointer at
data_dir()/checkpoints.db (short-term per-thread memory). Pass checkpointer=None
in GraphDeps to use an in-memory saver (tests).

LangSmith: node-level tracing is automatic when LANGSMITH_TRACING=true +
LANGSMITH_API_KEY are set in the env — LangGraph emits a run tree per node with
no code change here (openai calls are already wrapped in llm.py). Nothing to
import; the env var is the switch.
"""
from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from magenta.config import data_dir
from magenta.graph import nodes as N
from magenta.graph.state import OverallState
from magenta.memory.store import CustomerMemory


@dataclass
class GraphDeps:
    risk: object
    uplift: object
    bandit: object
    catalog: object
    oracle: object
    conn: object
    params: object
    chat: object
    load_customer: Callable[[str], object]
    checkpointer: object | None = None  # None => InMemorySaver (tests)
    campaign_id: str = "AGENT-EXP"
    system2_enabled: bool = False  # Lab 7 Task 7.6 wires System-2; "agent" rung sets True
    memory: CustomerMemory | None = None  # Lab 12: temporal customer memory (optional)


def build_graph(deps: GraphDeps):
    g = StateGraph(OverallState)

    bind = lambda fn: functools.partial(fn, deps=deps)
    g.add_node("sense", bind(N.sense))
    g.add_node("diagnose", bind(N.diagnose))
    g.add_node("decide", bind(N.decide))
    g.add_node("guardrail", bind(N.guardrail))
    g.add_node("act", bind(N.act))
    g.add_node("outcome", bind(N.outcome))

    g.add_edge(START, "sense")
    g.add_conditional_edges("sense", N.should_engage,
                            {"diagnose": "diagnose", "END": END})
    g.add_edge("diagnose", "decide")
    g.add_edge("decide", "guardrail")
    g.add_conditional_edges("guardrail", N.guardrail_route,
                            {"act": "act", "END": END})
    g.add_edge("act", "outcome")
    g.add_edge("outcome", END)

    checkpointer = deps.checkpointer if deps.checkpointer is not None else InMemorySaver()
    return g.compile(checkpointer=checkpointer)


def open_sqlite_saver(path: str | Path | None = None):
    """Context manager returning a SqliteSaver. Use in the CLI.

    NOTE (brief bug fixed on sight): the brief's default was the cwd-relative
    literal "data/checkpoints.db", which only resolves correctly if the
    process cwd happens to be the repo root. Commands are documented as
    `cd backend && uv run magenta ...`, so cwd is `backend/` and that literal
    would silently create `backend/data/checkpoints.db` instead. Anchor
    through `magenta.config.data_dir()` (repo-root-relative) instead, matching
    the RiskModel/UpliftModel default-path convention used elsewhere.
    """
    db_path = Path(path) if path is not None else data_dir() / "checkpoints.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver.from_conn_string(str(db_path))


def persist_audit(conn, audit_log: list[dict]) -> None:
    """Flush the accumulated audit_log into AUDIT_LOG (one row per node run)."""
    for entry in audit_log:
        conn.execute(
            "INSERT INTO AUDIT_LOG (NODE, CUSTOMER_ID, TS, PAYLOAD) "
            "VALUES (?, ?, ?, ?)",
            (entry["NODE"], entry["CUSTOMER_ID"], entry["TS"],
             entry.get("PAYLOAD", json.dumps({}))),
        )
    conn.commit()
