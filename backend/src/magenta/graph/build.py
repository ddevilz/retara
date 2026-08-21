"""Assemble the retention decision graph (§5.5).

Topology (explicit StateGraph — NOT supervisor/swarm; compliance wants a legible
state machine):

    START -> sense
    sense --(engage?)--> diagnose | END        # engage-gate = cost firewall
    diagnose -> decide -> guardrail
    guardrail --(pass/needs_approval?)--> act | END   # REJECT stops here
    act -> outcome -> END

thread_id = f"{tenant_id}:{customer_id}:{campaign_id}". PostgresSaver checkpointer against
DATABASE_URL (short-term per-thread memory). Pass checkpointer=None in GraphDeps
to use an in-memory saver (tests).

LangSmith: node-level tracing is automatic when LANGSMITH_TRACING=true +
LANGSMITH_API_KEY are set in the env — LangGraph emits a run tree per node with
no code change here (openai calls are already wrapped in llm.py). Nothing to
import; the env var is the switch.
"""
from __future__ import annotations

import functools
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from sqlalchemy import text
from sqlalchemy.engine import Connection

from magenta.db import database_url
from magenta.graph import nodes as N
from magenta.graph.state import OverallState
from magenta.graph.tables import DEFAULT_TENANT_ID
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
    tenant_id: str = DEFAULT_TENANT_ID  # Phase 1.3's get_graph_deps(tenant_id) sets this
    system2_enabled: bool = False  # Lab 7 Task 7.6 wires System-2; "agent" rung sets True
    memory: CustomerMemory | None = None  # Lab 12: temporal customer memory (optional)


def build_graph(deps: GraphDeps):
    g = StateGraph(OverallState)

    def bind(fn):
        return functools.partial(fn, deps=deps)

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

    # deps.checkpointer is typed `object` (GraphDeps keeps its fields loosely
    # duck-typed for dependency injection); callers only ever put a real
    # BaseCheckpointSaver there (open_postgres_saver()) or leave it None.
    checkpointer = (cast(BaseCheckpointSaver, deps.checkpointer)
                    if deps.checkpointer is not None else InMemorySaver())
    return g.compile(checkpointer=checkpointer)


@contextmanager
def open_postgres_saver():
    """Context manager yielding a PostgresSaver against DATABASE_URL.

    LangGraph checkpointer. One caller: the `magenta run-one` CLI command.
    Every other path leaves GraphDeps.checkpointer None and gets InMemorySaver.

    setup() is idempotent (CREATE TABLE IF NOT EXISTS internally) and creates
    LangGraph's own checkpoint tables, which Alembic does not own -- without
    it, the first run against a fresh database fails with UndefinedTable.

    `database_url()` is a SQLAlchemy DSN (`postgresql+psycopg://...`), but
    `PostgresSaver` connects with raw `psycopg`, which doesn't understand the
    `+psycopg` driver suffix in the scheme (`psycopg.ProgrammingError: missing
    "=" after "postgresql+psycopg://..."`) -- strip it to the plain
    `postgresql://` scheme psycopg expects.
    """
    conn_string = database_url().replace("postgresql+psycopg://", "postgresql://", 1)
    with PostgresSaver.from_conn_string(conn_string) as saver:
        saver.setup()
        yield saver


def persist_audit(conn: Connection, tenant_id: str, audit_log: list[dict]) -> None:
    """Flush the accumulated audit_log into AUDIT_LOG (one row per node run).

    `nodes._audit()` already json.dumps()s PAYLOAD once and stamps TS as an ISO
    string (audit_log entries flow through LangGraph's operator.add state
    reducer, so they must stay plain JSON-safe values, not psycopg-specific
    types). Re-applying json.dumps() here would double-encode PAYLOAD into a
    JSON string instead of a JSON object, so it's passed straight through and
    CAST to jsonb; TS is parsed back into a datetime for the TIMESTAMPTZ column.
    """
    for entry in audit_log:
        conn.execute(
            text(
                'INSERT INTO "AUDIT_LOG" ("TENANT_ID", "NODE", "CUSTOMER_ID", "TS", "PAYLOAD") '
                "VALUES (:tenant_id, :node, :customer_id, :ts, CAST(:payload AS jsonb))"
            ),
            {
                "tenant_id": tenant_id,
                "node": entry["NODE"],
                "customer_id": entry["CUSTOMER_ID"],
                "ts": datetime.fromisoformat(entry["TS"]),
                "payload": entry.get("PAYLOAD", "{}"),
            },
        )
    conn.commit()
