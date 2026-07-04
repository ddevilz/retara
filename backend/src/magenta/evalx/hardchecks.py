"""Hard invariant scans -- Task 9.3 (spec §0.5 build contract, compliance rails).

Reusable, deterministic pass/fail scan functions backing the `hardcheck`
pytest marker (`uv run pytest -m hardcheck`). These centralize FOUR
invariants that were previously asserted only piecemeal inside individual
graph node tests (tests/graph/test_nodes.py, test_build.py):

1. idempotency   -- replaying Act never produces more than one FULFILLMENTS
                    row per customer (`replay_idempotent`).
2. holdout purity -- a customer the graph itself shadow-logged as holdout
                    (ACT audit entry, status=SHADOW) must never ALSO carry a
                    real FULFILLMENTS row (`scan_holdout_purity`).
3. guardrail compliance -- every real fulfillment (ACT audit entry, status
                    FULFILLED/IDEMPOTENT_HIT) must be backed by a PASS or
                    NEEDS_APPROVAL GUARDRAIL verdict for that customer
                    (`scan_guardrail_compliance`).
4. no hidden leak -- L1 latent fields (owned exclusively by the simulator's
                    HiddenStore, CLAUDE.md anti-circularity rule) never
                    survive serialization of graph/dialogue state
                    (`scan_hidden_leak`).

Schema note: the task brief's illustrative SQL assumed a HOLDOUT column on
FULFILLMENTS and EVENT/VERDICT columns on AUDIT_LOG. Neither exists in the
real labs 0-7 schema (`magenta.graph.tables`): FULFILLMENTS never gets a row
at all for a holdout customer -- `graph.nodes.act` branches to a SHADOW audit
entry instead of inserting -- and AUDIT_LOG stores one JSON PAYLOAD blob per
(NODE, CUSTOMER_ID) event. The scans below read the real ALL_CAPS columns
(NODE / CUSTOMER_ID / PAYLOAD) and inspect the JSON payload for the
status/decision the corresponding node actually writes (see
`magenta.graph.nodes._audit`, `.act`, `.guardrail`). The scan LOGIC (a
violation = fulfilled-but-holdout, or fulfilled-without-PASS) is unchanged
from the brief -- only the column/table plumbing is adapted.
"""
from __future__ import annotations

import json
import sqlite3

## Hidden L1 field names that must NEVER appear in graph/dialogue state or
## prompts. Mirrors CLAUDE.md's anti-circularity list, magenta.sim.population's
## HiddenStore fields, and magenta.chat.persona's private CustomerBrief fields.
_HIDDEN_FIELDS = {
    "theta_churn_base", "theta_price_sens", "persuadable_segment",
    "competitor_pull", "accept_threshold_eur", "bluff", "brief_text", "true_cause",
}

_FULFILLED_STATUSES = {"FULFILLED", "IDEMPOTENT_HIT"}
_APPROVED_DECISIONS = {"PASS", "NEEDS_APPROVAL"}


def scan_hidden_leak(state_obj) -> list[str]:
    """Hidden-field names found anywhere in a serialized graph/dialogue state.

    Dumb-and-safe by design: stringify the WHOLE blob and substring-match
    every hidden field name, case-insensitively. A structural allow-list scan
    (checking only top-level keys) would miss a leak nested inside a
    narrative string, a prompt, or a list -- exactly the shape a leak takes
    when it happens in this codebase (LLM narrative text, not a raw field).
    """
    blob = json.dumps(state_obj, default=str).lower()
    return sorted(f for f in _HIDDEN_FIELDS if f.lower() in blob)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None


def _payloads_for_node(conn: sqlite3.Connection, node: str) -> list[tuple[str, dict]]:
    """[(customer_id, payload_dict), ...] for every AUDIT_LOG row of that NODE.

    Malformed/missing PAYLOAD is skipped rather than raising -- a compliance
    scan must never crash on the very audit trail it's auditing.
    """
    rows = conn.execute(
        "SELECT CUSTOMER_ID, PAYLOAD FROM AUDIT_LOG WHERE NODE = ?", (node,)
    ).fetchall()
    out: list[tuple[str, dict]] = []
    for cid, payload in rows:
        try:
            data = json.loads(payload)
        except (TypeError, ValueError):
            continue
        if isinstance(data, dict):
            out.append((cid, data))
    return out


def scan_holdout_purity(conn: sqlite3.Connection) -> list[str]:
    """customer_ids present in FULFILLMENTS that the audit trail ALSO recorded
    as holdout (an ACT-node SHADOW entry).

    These two signals are mutually exclusive by construction in
    `graph.nodes.act`: a holdout customer_id gets a SHADOW audit entry and NO
    FULFILLMENTS row; a real fulfillment gets a FULFILLED/IDEMPOTENT_HIT audit
    entry and a FULFILLMENTS row. Their agreement is a purity violation (e.g.
    a re-run with a flipped holdout flag actually charged/contacted someone).
    Empty conn / missing tables => vacuously pure (returns []).
    """
    if not _table_exists(conn, "FULFILLMENTS") or not _table_exists(conn, "AUDIT_LOG"):
        return []
    fulfilled_ids = {
        r[0] for r in conn.execute(
            "SELECT DISTINCT CUSTOMER_ID FROM FULFILLMENTS").fetchall()
    }
    if not fulfilled_ids:
        return []
    shadow_ids = {
        cid for cid, payload in _payloads_for_node(conn, "ACT")
        if payload.get("status") == "SHADOW"
    }
    return sorted(fulfilled_ids & shadow_ids)


def scan_guardrail_compliance(conn: sqlite3.Connection) -> list[str]:
    """customer_ids with a real fulfillment (AUDIT_LOG ACT status FULFILLED or
    IDEMPOTENT_HIT) but no matching PASS/NEEDS_APPROVAL GUARDRAIL verdict for
    that customer -- i.e. a fulfillment that bypassed (or outran) the
    compliance gate. Empty conn / missing table => vacuously compliant.
    """
    if not _table_exists(conn, "AUDIT_LOG"):
        return []
    fulfilled_ids = {
        cid for cid, payload in _payloads_for_node(conn, "ACT")
        if payload.get("status") in _FULFILLED_STATUSES
    }
    if not fulfilled_ids:
        return []
    approved_ids = {
        cid for cid, payload in _payloads_for_node(conn, "GUARDRAIL")
        if payload.get("decision") in _APPROVED_DECISIONS
    }
    return sorted(fulfilled_ids - approved_ids)


def replay_idempotent(conn: sqlite3.Connection, customer_id: str) -> bool:
    """True iff FULFILLMENTS holds at most one row for customer_id.

    This is the invariant `graph.nodes.act`'s insert-or-return (keyed on
    IDEMPOTENCY_KEY) is supposed to guarantee even when Act re-runs after an
    interrupt/resume (see
    tests/graph/test_nodes.py::test_act_fulfills_once_on_double_invoke, and
    this module's own test_replay_idempotent_after_double_act_invoke).
    Missing table => vacuously true (nothing has been fulfilled yet).
    """
    if not _table_exists(conn, "FULFILLMENTS"):
        return True
    n = conn.execute(
        "SELECT COUNT(*) FROM FULFILLMENTS WHERE CUSTOMER_ID = ?", (customer_id,)
    ).fetchone()[0]
    return n <= 1
