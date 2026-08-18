"""Read endpoints: scorecards, customer directory + 360, audit trail."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from magenta.api import data_access as da
from magenta.api.schemas import (
    AuditRow,
    Customer360,
    CustomerSummary,
    Scorecards,
)
from magenta.graph.tables import DEFAULT_TENANT_ID

router = APIRouter(prefix="/api", tags=["data"])


@router.get("/scorecards", response_model=Scorecards)
def scorecards() -> Scorecards:
    return da.load_scorecards()


@router.get("/customers", response_model=list[CustomerSummary])
def customers(
    limit: int = Query(50, ge=1, le=500),
    search: str = Query(""),
) -> list[CustomerSummary]:
    return da.list_customers(limit=limit, search=search)


@router.get("/customers/{customer_id}", response_model=Customer360)
def customer_360(customer_id: str) -> Customer360:
    c = da.get_customer(customer_id)
    if c is None:
        raise HTTPException(status_code=404, detail=f"unknown customer {customer_id}")
    return Customer360(customer=c, audit=da.audit_rows(DEFAULT_TENANT_ID, customer_id))


@router.get("/audit", response_model=list[AuditRow])
def audit(customer_id: str = Query(...)) -> list[AuditRow]:
    return da.audit_rows(DEFAULT_TENANT_ID, customer_id)
