"""System-2 deliberation for the high-value / ambiguous tail (§5.2).

Routed here only when clv_estimate >= P90 or diagnosis.confidence < 0.5 — the
router is the cost-engineering signal (§5.2: cannot run a 5-agent council on
360k customers). Kept to 3-4 large-role LLM calls:

  council (2 parallel lenses) -> planner (merge, no LLM) ->
  lookahead (uplift x bandit posterior, no LLM) -> critic (1 call).

Lookahead plans against the LEARNED surrogate (uplift.tau + bandit posterior),
NEVER the simulator's hidden oracle -> no circularity (§5.2 centerpiece).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from magenta.brain.features import featurize
from magenta.graph.state import Diagnosis, RiskUpliftReport
from magenta.offers import Arm, OfferDecision


def should_deliberate(customer, diagnosis: Diagnosis, p90_clv: float) -> bool:
    high_value = getattr(customer, "clv_estimate", 0.0) >= p90_clv
    ambiguous = diagnosis.confidence < 0.5
    return bool(high_value or ambiguous)


_BILLING_LENS = (
    "You are the BILLING specialist on a retention council. Given the drivers and "
    "diagnosis, name the single best offer arm (one of the eligible arm ids). "
    "Answer with ONLY the arm id."
)
_NETWORK_LENS = (
    "You are the NETWORK/QoS specialist on a retention council. Given the drivers "
    "and diagnosis, name the single best offer arm (one of the eligible arm ids). "
    "Answer with ONLY the arm id."
)
_CRITIC = (
    "You are the retention CRITIC. Validate the proposed arm against policy, margin, "
    "and brand. Reply with exactly PASS or REJECT."
)


def _lens_prompt(system: str, report: RiskUpliftReport, diagnosis: Diagnosis,
                 eligible: list[Arm]) -> list[dict]:
    drivers = "; ".join(f"{d.label} ({d.direction})" for d in report.drivers)
    ids = ", ".join(a.value for a in eligible)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content":
            f"Drivers: {drivers}. Diagnosis: {diagnosis.narrative}. "
            f"Eligible arm ids: {ids}."},
    ]


def _parse_arm(text: str, eligible: list[Arm]) -> Arm | None:
    up = (text or "").strip().upper()
    for a in eligible:
        if a.value in up:
            return a
    return None


def deliberate(customer, report: RiskUpliftReport, diagnosis: Diagnosis,
               deps) -> OfferDecision:
    eligible = [a for a in deps.catalog.eligible(customer)
                if a.value in set(diagnosis.eligible_offer_ids)]
    if not eligible:
        eligible = [Arm.NO_ACTION]

    # 1) COUNCIL — 2 parallel large calls.
    with ThreadPoolExecutor(max_workers=2) as ex:
        billing_f = ex.submit(deps.chat.chat, "large",
                              _lens_prompt(_BILLING_LENS, report, diagnosis, eligible))
        network_f = ex.submit(deps.chat.chat, "large",
                              _lens_prompt(_NETWORK_LENS, report, diagnosis, eligible))
        candidates = [_parse_arm(billing_f.result(), eligible),
                      _parse_arm(network_f.result(), eligible)]

    # 2) PLANNER — merge council candidates ∩ eligible; fall back to all eligible.
    merged = [a for a in eligible if a in candidates] or eligible

    # 3) LOOKAHEAD — score via learned surrogate (uplift τ × bandit posterior).
    x = featurize(customer)
    tau = deps.uplift.tau(customer)
    best = max(merged, key=lambda a: tau * deps.bandit.posterior_mean(x, a))

    # 4) CRITIC — 1 large call validates the winner.
    critic_msgs = [
        {"role": "system", "content": _CRITIC},
        {"role": "user", "content":
            f"Proposed arm: {best.value}. Diagnosis: {diagnosis.narrative}. "
            "Validate."},
    ]
    verdict = (deps.chat.chat("large", critic_msgs) or "").strip().upper()
    if "REJECT" in verdict:
        best = min(merged, key=lambda a: deps.catalog.cost(a))  # cheapest fallback

    return OfferDecision(
        arm=best,
        cost=deps.catalog.cost(best),
        rationale=f"system2: {diagnosis.narrative}",
        propensity=1.0,
    )
