"""Cohort-level batched diagnosis with a driver-signature cache (§0.5).

Cost lever: customers whose SHAP driver *shape* is identical get the SAME
diagnosis, so we make one cheap LLM call per DISTINCT signature instead of one
per customer. At 50k customers this collapses to a few hundred calls.

Batch-API upgrade note
----------------------
This uses a ThreadPool of synchronous cheap-role calls (simple, good for the
demo). For production cohort scale, swap `_diagnose_one` for the OpenAI **Batch
API**: write one JSONL request per distinct signature, submit, poll, map results
back by signature. Same signature-cache logic, ~50% cost + no rate-limit risk.
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor

from magenta.graph.nodes import _DIAGNOSE_SYSTEM, _diagnose_user_prompt, _observables
from magenta.graph.state import Diagnosis, RiskUpliftReport


def driver_signature(report: RiskUpliftReport) -> str:
    parts = [report.band.value]
    for d in sorted(report.drivers, key=lambda x: x.feature):
        sign = "+" if d.shap_value >= 0 else "-"
        parts.append(f"{d.feature}|{sign}|{d.direction}")
    raw = ";".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _diagnose_one(chat, report: RiskUpliftReport, observables: dict) -> Diagnosis:
    messages = [
        {"role": "system", "content": _DIAGNOSE_SYSTEM},
        {"role": "user", "content": _diagnose_user_prompt(report, observables)},
    ]
    return chat.chat_structured("cheap", messages, Diagnosis)


def diagnose_cohort(customers, reports, chat, max_workers: int = 8) -> dict[str, Diagnosis]:
    by_id = {c.customer_id: c for c in customers}
    # one representative customer per distinct signature.
    sig_to_rep: dict[str, str] = {}
    id_to_sig: dict[str, str] = {}
    for cid, report in reports.items():
        if cid not in by_id:
            continue
        sig = driver_signature(report)
        id_to_sig[cid] = sig
        sig_to_rep.setdefault(sig, cid)

    def work(sig_cid):
        sig, cid = sig_cid
        report = reports[cid]
        obs = _observables(by_id[cid])
        return sig, _diagnose_one(chat, report, obs)

    sig_to_diag: dict[str, Diagnosis] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for sig, diag in ex.map(work, list(sig_to_rep.items())):
            sig_to_diag[sig] = diag

    return {cid: sig_to_diag[sig] for cid, sig in id_to_sig.items()}
