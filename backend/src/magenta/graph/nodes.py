"""LangGraph node functions for the retention decision spine (§5.5).

Every node takes (state, deps) and returns a PARTIAL state dict. build.py binds
`deps` via functools.partial so the compiled graph sees plain (state)->dict.

Contract guarantees enforced here:
- diagnose does exactly ONE cheap-role LLM call, over SHAP drivers + L2
  observables ONLY. No L1 latent field is ever read or serialized (asserted).
- every executed node appends exactly one AUDIT_LOG entry via operator.add.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

from magenta.brain.features import featurize
from magenta.brain.uplift import Segment, classify_segment
from magenta.graph import system2
from magenta.graph.state import (
    Diagnosis,
    GuardrailVerdict,
    OverallState,
    RiskUpliftReport,
    Timing,
)
from magenta.graph.tables import (
    contacts_since,
    fulfillment_for,
    insert_fulfillment,
    record_contact,
)
from magenta.offers import Arm, OfferDecision

## whitelist of L2 observable fields we are allowed to describe to the LLM.
_OBSERVABLE_FIELDS = (
    # MUST be real magenta.sim.population.Customer fields (drift-pinned by test).
    "tenure_months", "monthly_charge", "overage_events_90d", "dropped_calls_30d",
    "support_tickets_90d", "contract_end_days", "gross_margin_monthly", "clv_estimate",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _audit(node: str, customer_id: str, payload: dict) -> dict:
    return {
        "NODE": node,
        "CUSTOMER_ID": customer_id,
        "TS": _now_iso(),
        "PAYLOAD": json.dumps(payload, default=str),
    }


def _observables(customer) -> dict:
    """Extract ONLY whitelisted L2 fields. Structurally cannot leak L1."""
    return {f: getattr(customer, f, None) for f in _OBSERVABLE_FIELDS}


## --------------------------------------------------------------------------- #
## 1. SENSE — ML brain, no LLM. The engage-gate + cost firewall.
## --------------------------------------------------------------------------- #
def sense(state: OverallState, deps) -> dict:
    customer = deps.load_customer(state["customer_id"])
    assessment = deps.risk.score(customer)
    tau = deps.uplift.tau(customer)
    segment = classify_segment(assessment.p_churn, tau)
    # engage only for PERSUADABLE (never sure-thing/lost-cause/sleeping-dog).
    engage = segment is Segment.PERSUADABLE
    timing = Timing.ACT_NOW if assessment.band.value in ("HIGH", "CRITICAL") else Timing.SNOOZE
    report = RiskUpliftReport(
        p_churn=assessment.p_churn,
        band=assessment.band,
        drivers=assessment.drivers,
        tau_hat=tau,
        segment=segment,
        engage=engage,
        timing=timing,
    )
    payload = {"p_churn": report.p_churn, "segment": segment.value,
               "tau_hat": tau, "engage": engage, "timing": timing.value}
    return {"risk": report,
            "audit_log": [_audit("SENSE", state["customer_id"], payload)]}


def should_engage(state: OverallState) -> str:
    """Conditional edge from sense: 'diagnose' if engaged else 'END'."""
    r = state.get("risk")
    return "diagnose" if (r is not None and r.engage) else "END"


## --------------------------------------------------------------------------- #
## 2. DIAGNOSE — exactly ONE cheap LLM call over SHAP drivers + observables.
## --------------------------------------------------------------------------- #
_ARM_MENU = (
    "NO_ACTION: no contact; "
    "ACKNOWLEDGE_AND_FIX: apologize + resolve service complaints/dropped calls; "
    "BILL_CREDIT: one-time credit for bill shock/overage; "
    "PLAN_DOWNSELL: cheaper better-fitting plan for price pain; "
    "DATA_BOOST: extra data for overage pain; "
    "DEVICE_UPGRADE: device voucher near contract end; "
    "NETWORK_PRIORITY_FIX: priority network ticket for coverage pain; "
    "BUNDLE_ADDON: bundle sweetener vs competitor pull/contract end"
)

_DIAGNOSE_SYSTEM = (
    "You are a telecom retention analyst. Given a customer's observable account "
    "signals and the churn model's SHAP drivers, name the root-cause tags, write a "
    "one-paragraph narrative, and list which offer types could fit. Base your answer "
    "ONLY on the signals provided. Do not invent internal scores. "
    "eligible_offer_ids MUST be chosen from exactly this menu (verbatim ids): "
    + _ARM_MENU
)


def _diagnose_user_prompt(report: RiskUpliftReport, observables: dict) -> str:
    drivers = "; ".join(
        f"{d.label} (shap={d.shap_value:+.2f}, dir={d.direction})"
        for d in report.drivers
    )
    obs = "; ".join(f"{k}={v}" for k, v in observables.items() if v is not None)
    return (
        f"Churn probability band: {report.band.value}.\n"
        f"Top SHAP drivers: {drivers}.\n"
        f"Observable account signals: {obs}.\n"
        "Return root_cause_tags, narrative, eligible_offer_ids, confidence. "
        "root_cause_tags MUST be chosen from exactly: OVERAGE, DROPPED_CALLS, "
        "BILL_SHOCK, COMPETITOR_OFFER, CONTRACT_EXPIRY, SERVICE_COMPLAINT "
        "(the offer catalog matches on these verbatim ids)."
    )


def diagnose(state: OverallState, deps) -> dict:
    customer = deps.load_customer(state["customer_id"])
    report = state["risk"]
    observables = _observables(customer)
    messages = [
        {"role": "system", "content": _DIAGNOSE_SYSTEM},
        {"role": "user", "content": _diagnose_user_prompt(report, observables)},
    ]
    try:
        diagnosis: Diagnosis = deps.chat.chat_structured("cheap", messages, Diagnosis)
    except Exception as exc:  # degraded diagnosis: empty arm set -> NO_ACTION path
        diagnosis = Diagnosis(root_cause_tags=["DIAGNOSIS_FAILED"],
                              narrative=f"LLM diagnosis failed: {type(exc).__name__}",
                              eligible_offer_ids=[], confidence=0.0)
    payload = {"root_cause_tags": diagnosis.root_cause_tags,
               "eligible_offer_ids": diagnosis.eligible_offer_ids,
               "confidence": diagnosis.confidence}
    return {"diagnosis": diagnosis,
            "audit_log": [_audit("DIAGNOSE", state["customer_id"], payload)]}


## --------------------------------------------------------------------------- #
## 3. DECIDE — bandit pick over (catalog.eligible ∩ diagnosis.eligible_offer_ids)
## --------------------------------------------------------------------------- #
def decide(state: OverallState, deps) -> dict:
    customer = deps.load_customer(state["customer_id"])
    diagnosis = state["diagnosis"]
    catalog_eligible = deps.catalog.eligible(customer)
    allowed_ids = set(diagnosis.eligible_offer_ids)
    eligible = [a for a in catalog_eligible if a.value in allowed_ids]
    if not eligible:
        eligible = [Arm.NO_ACTION]

    if (getattr(deps, "system2_enabled", False)
            and system2.should_deliberate(customer, diagnosis, deps.params.p90_clv)):
        offer = system2.deliberate(customer, state.get("risk"), diagnosis, deps)
        payload = {"arm": offer.arm.value, "cost": offer.cost, "path": "SYSTEM2",
                   "eligible": [a.value for a in eligible]}
        return {"offer": offer,
                "audit_log": [_audit("DECIDE_S2", state["customer_id"], payload)]}

    x = featurize(customer)
    arm, propensity = deps.bandit.select(x, eligible)
    offer = OfferDecision(
        arm=arm,
        cost=deps.catalog.cost(arm),
        rationale=diagnosis.narrative,
        propensity=propensity,
    )
    payload = {"arm": arm.value, "cost": offer.cost, "propensity": propensity,
               "eligible": [a.value for a in eligible]}
    return {"offer": offer,
            "audit_log": [_audit("DECIDE", state["customer_id"], payload)]}


## --------------------------------------------------------------------------- #
## 4. GUARDRAIL — deterministic, fails closed. No protected attrs. (§5.7)
## --------------------------------------------------------------------------- #
def guardrail(state: OverallState, deps) -> dict:
    offer = state["offer"]
    customer = deps.load_customer(state["customer_id"])
    failed: list[str] = []
    requires_approval = False

    if offer is None or offer.arm is Arm.NO_ACTION:
        verdict = GuardrailVerdict(decision="PASS", failed_policies=[])
        return {"verdict": verdict,
                "audit_log": [_audit("GUARDRAIL", state["customer_id"],
                                     {"decision": "PASS", "reason": "NO_ACTION"})]}

    # 1) consent
    if not state["consent_flags"].get("MARKETING"):
        failed.append("CONSENT")

    # 2) frequency cap (Store/ledger)
    since = (datetime.now(timezone.utc)
             - timedelta(days=deps.params.freq_cap_days)).isoformat()
    if contacts_since(deps.conn, state["customer_id"], since) >= deps.params.freq_cap_max:
        failed.append("FREQ_CAP")

    # 3) min-margin: post-offer margin must clear the arm's floor
    # NOTE: brief snippet used getattr(customer, "gross_margin", 0.0) — that
    # field doesn't exist on the real Customer model (magenta.sim.population),
    # so it silently always evaluated to 0.0. The real field is
    # gross_margin_monthly; use direct attribute access (no getattr-default)
    # so a future rename fails loudly instead of silently defaulting.
    margin_after = customer.gross_margin_monthly - offer.cost
    if margin_after < deps.catalog.min_margin(offer.arm):
        failed.append("MIN_MARGIN")

    # 4) value cap -> human approval (interrupt() wired in lab 8; keep idempotent)
    if offer.cost > deps.params.value_cap:
        requires_approval = True

    if failed:
        decision = "REJECT"
    elif requires_approval:
        decision = "NEEDS_APPROVAL"
    else:
        decision = "PASS"

    verdict = GuardrailVerdict(decision=decision, failed_policies=failed)
    payload = {"decision": decision, "failed_policies": failed,
               "requires_approval": requires_approval}
    return {"verdict": verdict,
            "requires_approval": requires_approval,
            "audit_log": [_audit("GUARDRAIL", state["customer_id"], payload)]}


def guardrail_route(state: OverallState) -> str:
    """Conditional edge: 'act' on PASS/NEEDS_APPROVAL, 'END' on REJECT."""
    v = state.get("verdict")
    if v is None or v.decision == "REJECT":
        return "END"
    return "act"


## --------------------------------------------------------------------------- #
## 5. ACT — idempotent fulfillment; holdout ⇒ shadow-log only. (§5.5 / risk #4)
## --------------------------------------------------------------------------- #
def idempotency_key(customer_id: str, campaign_id: str, arm: Arm) -> str:
    raw = f"{customer_id}:{campaign_id}:{arm.value}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def act(state: OverallState, deps) -> dict:
    offer = state["offer"]
    cid, camp = state["customer_id"], state["campaign_id"]

    if offer is None or offer.arm is Arm.NO_ACTION:
        return {"fulfillment": {"status": "NO_ACTION"},
                "audit_log": [_audit("ACT", cid, {"status": "NO_ACTION"})]}

    if state["holdout"]:
        # shadow: record the counterfactual, fulfill nothing, no contact ledger.
        shadow = {"status": "SHADOW", "arm": offer.arm.value, "cost": offer.cost,
                  "idempotency_key": idempotency_key(cid, camp, offer.arm)}
        return {"fulfillment": shadow,
                "audit_log": [_audit("ACT", cid, {"status": "SHADOW",
                                                  "arm": offer.arm.value})]}

    key = idempotency_key(cid, camp, offer.arm)
    already = fulfillment_for(deps.conn, key) is not None
    row = insert_fulfillment(deps.conn, key, cid, camp, offer.arm.value,
                             offer.cost, "FULFILLED")
    if not already:
        record_contact(deps.conn, cid, camp, _now_iso())
    return {"fulfillment": row,
            "audit_log": [_audit("ACT", cid,
                                 {"status": "FULFILLED" if not already else "IDEMPOTENT_HIT",
                                  "arm": offer.arm.value, "key": key})]}


## --------------------------------------------------------------------------- #
## 6. OUTCOME — oracle result → reward → bandit.update → audit. (§5.5)
## --------------------------------------------------------------------------- #
def outcome(state: OverallState, deps) -> dict:
    customer = deps.load_customer(state["customer_id"])
    offer = state["offer"]
    holdout = state["holdout"]
    no_action = offer is None or offer.arm is Arm.NO_ACTION

    # holdout measures the counterfactual: oracle sees NO offer.
    oracle_offer = None if (holdout or no_action) else offer
    result = deps.oracle.outcome(customer, oracle_offer)
    retained = not result.churned
    # NOTE: brief snippet used getattr(customer, "gross_margin", 0.0) — that
    # field doesn't exist on the real Customer model (magenta.sim.population),
    # so it silently always evaluated to 0.0. The real field is
    # gross_margin_monthly; use direct attribute access (no getattr-default)
    # so a future rename fails loudly instead of silently defaulting.
    # Annualized margin (x12) — MUST match the Lab-5 bandit-episodes reward scale
    # (retained*margin*12 - cost, per the plan's ML contract): the same bandit
    # posterior is updated from both paths, so mixed scales would corrupt it.
    margin_annual = customer.gross_margin_monthly * 12.0
    cost = 0.0 if no_action else offer.cost
    reward = (margin_annual if retained else 0.0) - cost

    if not holdout and not no_action:
        deps.bandit.update(featurize(customer), offer.arm, reward)

    out = {"accepted": bool(result.accepted), "churned": bool(result.churned),
           "retained": retained, "reward": reward, "holdout": holdout}
    return {"outcome": out,
            "audit_log": [_audit("OUTCOME", state["customer_id"], out)]}
