"""Cohort-level batched diagnosis with a semantic driver-shape cache (§0.5).

Cost lever: customers whose SHAP driver *shape* is identical -- or close
enough in embedding space -- get the SAME diagnosis text, so a large cohort
run makes far fewer LLM calls than one per customer. At 50k customers this
collapses to a few hundred calls.

Task 13.4 upgrade: this used to be an in-process exact-hash cache
(`driver_signature`, still exposed below for callers/audit that want a
stable identity for a driver shape). It is now backed by an optional
`magenta.cost.cache.SemanticCache` (persisted in `conn`, cosine similarity
over `_diagnosis_key_text`), routed through `magenta.cost.cascade.cascade`
(cheap-first, escalate to large on low confidence) and metered by an
optional `magenta.cost.meter.CostMeter`. Passing neither `cache` nor `meter`
keeps this a plain per-customer cascade with no cost bookkeeping.

Batch-API upgrade note
----------------------
This calls `_chat` synchronously per customer (simple, good for the demo).
For production cohort scale, swap `_chat` for the OpenAI **Batch API**: write
one JSONL request per cache-miss key-text, submit, poll, map results back.
Same cache logic, ~50% cost + no rate-limit risk.
"""
from __future__ import annotations

import hashlib

from magenta.cost.cascade import cascade
from magenta.graph.nodes import _DIAGNOSE_SYSTEM, _diagnose_user_prompt, _observables
from magenta.graph.state import RiskUpliftReport
from magenta.llm import chat


def driver_signature(report: RiskUpliftReport) -> str:
    """Stable exact-hash identity for a driver shape (band + sorted
    feature/sign/direction). Kept for callers that want a deterministic key
    (audit logs, dedup diagnostics) independent of the embedder."""
    parts = [report.band.value]
    for d in sorted(report.drivers, key=lambda x: x.feature):
        sign = "+" if d.shap_value >= 0 else "-"
        parts.append(f"{d.feature}|{sign}|{d.direction}")
    raw = ";".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _diagnosis_key_text(report: RiskUpliftReport) -> str:
    """Natural-language SemanticCache key: band + segment + SHAP driver
    shape. Two customers whose driver shapes are identical embed to cosine
    1.0 (guaranteed cache hit); customers whose shapes are merely SIMILAR
    (e.g. same primary driver plus a minor secondary one, not byte-identical)
    can still clear the cache's cosine threshold -- the upgrade
    `driver_signature`'s exact hash could never give.

    KNOWN LIMITATION (verified on this machine): a general-purpose sentence
    encoder (all-MiniLM-L6-v2) barely moves for a single antonym swap in an
    otherwise-identical short technical phrase -- "OVERAGE_EVENTS ... UP"
    vs "...DOWN" for the SAME feature still scores ~0.97 cosine, statistically
    indistinguishable from a genuine magnitude-only near-dup. A sign/direction
    flip on an otherwise-identical driver text can therefore be mis-treated as
    a near-duplicate. This is a real, accepted cost/quality tradeoff of using
    embedding similarity here -- it is exactly what `magenta cost report`'s
    quality_retained metric (cascade vs forced-large agreement) is meant to
    surface, not something this cache can structurally rule out."""
    driver_desc = "; ".join(
        f"{d.label} shows a {'positive' if d.shap_value >= 0 else 'negative'} "
        f"{d.direction.lower()} impact"
        for d in sorted(report.drivers, key=lambda x: x.feature)
    )
    return f"churn risk band {report.band.value}, segment {report.segment.value}; drivers: {driver_desc}"


def _chat(role: str, messages: list[dict]) -> str:
    """Bare module-level chat_fn for `cascade` -- monkeypatchable directly in
    tests without threading a fake client through GraphDeps. Delegates to
    `magenta.llm.chat` (plain text; the cohort cost lever only needs a short
    diagnosis string to cache and compare, not the full structured
    `Diagnosis` object the single-customer graph path builds)."""
    return chat(role, messages)


_HEDGE_WORDS = ("unsure", "unknown", "not sure", "n/a", "unclear")


def _confidence_from_answer(answer: str) -> float:
    """Heuristic confidence signal for the cascade: a missing/very short or
    hedging answer is treated as low-confidence and escalated to the large
    role; anything else is treated as confident. (No learned verifier model
    in this lab -- see Task 13.5, a stretch item, skipped.)"""
    if not answer or len(answer.strip()) < 3:
        return 0.0
    lowered = answer.lower()
    if any(w in lowered for w in _HEDGE_WORDS):
        return 0.3
    return 0.9


def diagnose_cohort(customers, reports, deps, meter=None, cache=None) -> dict[str, str]:
    """One cascade(cheap->large) diagnosis per customer, short-circuited by
    an optional `SemanticCache` keyed on driver shape + segment.

    `deps` is accepted for interface parity with the single-customer
    `nodes.diagnose` (a GraphDeps/memory hook for future wiring) -- this
    cohort path does not read from it today.

    Returns `{customer_id: diagnosis_text}`. `meter`, if given, records every
    decision (cache hit, or the cascade's role/escalation outcome) so
    `magenta cost report` can print pct_routed_cheap / cache_hit_rate /
    escalation_rate.
    """
    by_id = {c.customer_id: c for c in customers}
    out: dict[str, str] = {}

    for cid, report in reports.items():
        customer = by_id.get(cid)
        if customer is None:
            continue

        key_text = _diagnosis_key_text(report)
        cached = cache.get(key_text) if cache is not None else None
        if cached is not None:
            if meter is not None:
                meter.record("cache", cache_hit=True, escalated=False)
            out[cid] = cached
            continue

        observables = _observables(customer)
        messages = [
            {"role": "system", "content": _DIAGNOSE_SYSTEM},
            {"role": "user", "content": _diagnose_user_prompt(report, observables)},
        ]
        result = cascade(messages, _chat, _confidence_from_answer)
        if meter is not None:
            meter.record(result.role_used, cache_hit=False, escalated=result.escalated)
        if cache is not None:
            cache.put(key_text, result.answer)
        out[cid] = result.answer

    return out
