"""Read endpoints: scorecards, customer directory + 360, audit trail."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from magenta.api import data_access as da
from magenta.api.schemas import (
    AuditRow,
    Customer360,
    CustomerSummary,
    Scorecards,
)
from magenta.auth import TenantContext, current_tenant

router = APIRouter(prefix="/api", tags=["data"])


@router.get("/scorecards", response_model=Scorecards)
def scorecards(tenant: TenantContext = Depends(current_tenant)) -> Scorecards:
    # tenant is unused until Phase 1.3 makes the population per-tenant. Removing this
    # parameter removes authentication from this route.
    return da.load_scorecards()


@router.get("/customers", response_model=list[CustomerSummary])
def customers(
    limit: int = Query(50, ge=1, le=500),
    search: str = Query(""),
    tenant: TenantContext = Depends(current_tenant),
) -> list[CustomerSummary]:
    # tenant is unused until Phase 1.3 makes the population per-tenant. Removing this
    # parameter removes authentication from this route.
    return da.list_customers(limit=limit, search=search)


@router.get("/customers/{customer_id}", response_model=Customer360)
def customer_360(
    customer_id: str,
    tenant: TenantContext = Depends(current_tenant),
) -> Customer360:
    c = da.get_customer(customer_id)
    if c is None:
        raise HTTPException(status_code=404, detail=f"unknown customer {customer_id}")
    return Customer360(customer=c, audit=da.audit_rows(tenant.tenant_id, customer_id))


@router.get("/audit", response_model=list[AuditRow])
def audit(
    customer_id: str = Query(...),
    tenant: TenantContext = Depends(current_tenant),
) -> list[AuditRow]:
    return da.audit_rows(tenant.tenant_id, customer_id)
