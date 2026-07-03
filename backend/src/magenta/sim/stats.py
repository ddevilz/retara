"""Population summary statistics for `magenta sim generate --stats`."""

from __future__ import annotations

from collections import Counter

from magenta.sim.oracle import ResponseOracle
from magenta.sim.population import Customer, HiddenStore, Segment


def population_stats(
    customers: list[Customer], hidden: HiddenStore, oracle: ResponseOracle
) -> dict:
    n = len(customers)
    contract_counts = Counter(c.contract for c in customers)
    plan_counts = Counter(c.plan for c in customers)
    seg_counts = Counter(hidden[c.customer_id].persuadable_segment for c in customers)

    nps_missing = sum(1 for c in customers if c.nps_last is None)
    mean_charge = sum(c.monthly_charge for c in customers) / n
    mean_tenure = sum(c.tenure_months for c in customers) / n

    # churn base rate = P(churn | no action), averaged deterministically
    base_churn = sum(oracle.churn_prob(c, None) for c in customers) / n

    return {
        "n": n,
        "contract_mix": {k: contract_counts[k] / n for k in sorted(contract_counts)},
        "plan_mix": {k: plan_counts[k] / n for k in sorted(plan_counts)},
        "segment_mix": {s.value: seg_counts[s] / n for s in Segment},
        "nps_missing_rate": nps_missing / n,
        "mean_monthly_charge": mean_charge,
        "mean_tenure_months": mean_tenure,
        "churn_base_rate": base_churn,
    }


def format_stats(stats: dict) -> str:
    lines: list[str] = []
    lines.append(f"POPULATION (n={stats['n']})")
    lines.append("")
    lines.append("CONTRACT MIX")
    for k, v in stats["contract_mix"].items():
        lines.append(f"  {k:<16} {v:6.1%}")
    lines.append("PLAN MIX")
    for k, v in stats["plan_mix"].items():
        lines.append(f"  {k:<16} {v:6.1%}")
    lines.append("SEGMENT MIX")
    for k, v in stats["segment_mix"].items():
        lines.append(f"  {k:<16} {v:6.1%}")
    lines.append("")
    lines.append(f"NPS MISSING RATE   {stats['nps_missing_rate']:6.1%}")
    lines.append(f"MEAN MONTHLY CHARGE  {stats['mean_monthly_charge']:8.2f}")
    lines.append(f"MEAN TENURE (MONTHS) {stats['mean_tenure_months']:8.1f}")
    lines.append(f"CHURN BASE RATE    {stats['churn_base_rate']:6.1%}")
    return "\n".join(lines)
