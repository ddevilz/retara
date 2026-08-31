"""Company profile: read/update the org's onboarding profile.

INDUSTRY IS NULL is how the frontend knows onboarding isn't done -- see
0004_company_profile.py. The PUT rejects any industry but "telecom"
server-side: the frontend disables the other dropdown options, but a client
bypassing that must not be able to set a value nothing downstream can honor.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text

from magenta.api.schemas import OrgProfile, OrgProfileUpdate
from magenta.auth import TenantContext, bound_tenant
from magenta.db import get_conn
from magenta.logging_config import get_logger

router = APIRouter(prefix="/api/org", tags=["org"])
logger = get_logger(__name__)

SUPPORTED_INDUSTRIES = {"telecom"}


@router.get("/profile", response_model=OrgProfile)
def get_profile(tenant: TenantContext = Depends(bound_tenant)) -> OrgProfile:
    with get_conn() as conn:
        row = conn.execute(
            text(
                'SELECT "NAME", "INDUSTRY", "MONTHLY_TOKEN_BUDGET", '
                '"ADMIN_CONTACT_EMAIL" FROM "ORGANIZATIONS" WHERE "ID" = :id'
            ),
            {"id": tenant.tenant_id},
        ).mappings().one()
    return OrgProfile(
        name=row["NAME"],
        industry=row["INDUSTRY"],
        monthly_token_budget=row["MONTHLY_TOKEN_BUDGET"],
        admin_contact_email=row["ADMIN_CONTACT_EMAIL"],
    )


@router.put("/profile", response_model=OrgProfile)
def update_profile(
    body: OrgProfileUpdate, tenant: TenantContext = Depends(bound_tenant)
) -> OrgProfile:
    if tenant.role != "org:admin":
        raise HTTPException(status_code=403, detail="admin role required")
    if body.industry not in SUPPORTED_INDUSTRIES:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported industry {body.industry!r}; only telecom is available today",
        )
    with get_conn() as conn:
        conn.execute(
            text(
                'UPDATE "ORGANIZATIONS" SET "NAME" = :name, "INDUSTRY" = :industry, '
                '"MONTHLY_TOKEN_BUDGET" = :budget, "ADMIN_CONTACT_EMAIL" = :contact '
                'WHERE "ID" = :id'
            ),
            {
                "name": body.name,
                "industry": body.industry,
                "budget": body.monthly_token_budget,
                "contact": body.admin_contact_email,
                "id": tenant.tenant_id,
            },
        )
        conn.commit()
    logger.info("org.profile_updated", tenant_id=tenant.tenant_id, industry=body.industry)
    return get_profile(tenant)
