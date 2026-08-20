"""One per-tenant simulated population, replacing three separate @lru_cache(maxsize=1)
copies that each held the same global population.

The HiddenStore is kept here because the oracle needs it, and it is never read by
anything that serves an API response — see `test_hidden_state_never_reaches_summaries`.
"""
from __future__ import annotations

from dataclasses import dataclass

from magenta.api.schemas import CustomerSummary
from magenta.sim.population import Customer, HiddenStore, generate_population
from magenta.tenancy import BoundedTTLCache, tenant_seed

POP_N = 2000

# Populations are pure functions of (POP_N, tenant_seed) and never change, so the TTL
# is long. The deps cache in Task 4 holds trained models and expires far sooner.
POPULATION_CACHE = BoundedTTLCache(maxsize=16, ttl_seconds=3600)


@dataclass
class TenantPopulation:
    customers: dict[str, Customer]
    summaries: list[CustomerSummary]
    hidden: HiddenStore


def _to_summary(c: Customer) -> CustomerSummary:
    """Moved verbatim from data_access._customer_to_summary (data_access.py:71-100).

    Field names here are pinned to the real magenta.sim.population.Customer
    model (verified against source, not guessed): monthly_charge (singular),
    nps_last, support_tickets_90d, clv_estimate, gross_margin_monthly.
    data_util_ratio / dropped_call_rate don't exist as raw fields — derived
    the same way magenta.brain.features._data_util_ratio does for the
    former; dropped_call_rate is dropped_calls_30d normalized to a per-day
    rate (the raw field is a 30-day count, and the frontend renders this to
    3 decimal places, so a fractional per-day rate is the sensible reading).
    """
    allowance = c.data_allowance_gb
    data_util_ratio = 0.0 if not allowance or allowance <= 0 else c.data_gb_used_p50 / allowance
    dropped_call_rate = c.dropped_calls_30d / 30.0

    return CustomerSummary(
        customer_id=c.customer_id,
        tenure_months=c.tenure_months,
        contract=c.contract,
        monthly_charges=c.monthly_charge,
        total_charges=c.total_charges,
        data_util_ratio=data_util_ratio,
        dropped_call_rate=dropped_call_rate,
        nps=float(c.nps_last) if c.nps_last is not None else None,
        support_tickets=c.support_tickets_90d,
        contract_end_days=c.contract_end_days,
        clv=c.clv_estimate,
        gross_margin=c.gross_margin_monthly,
    )


def get_population(tenant_id: str) -> TenantPopulation:
    cached = POPULATION_CACHE.get(tenant_id)
    if cached is not None:
        return cached
    customers, hidden = generate_population(POP_N, seed=tenant_seed(tenant_id))
    pop = TenantPopulation(
        customers={c.customer_id: c for c in customers},
        summaries=[_to_summary(c) for c in customers],
        hidden=hidden,
    )
    POPULATION_CACHE.put(tenant_id, pop)
    return pop


def list_customers(tenant_id: str, limit: int = 50, search: str = "") -> list[CustomerSummary]:
    rows = get_population(tenant_id).summaries
    if search:
        needle = search.lower()
        rows = [c for c in rows if needle in c.customer_id.lower()]
    return rows[:limit]


def get_customer(tenant_id: str, customer_id: str) -> CustomerSummary | None:
    return next(
        (c for c in get_population(tenant_id).summaries if c.customer_id == customer_id),
        None,
    )
