"""Company profile fields for tenant onboarding.

INDUSTRY IS NULL is the "profile incomplete" signal -- no separate boolean
flag, matching the existing MONTHLY_TOKEN_BUDGET IS NULL = "unconfigured"
convention (0003_llm_usage.py).

Revision ID: 0004
"""
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute('ALTER TABLE "ORGANIZATIONS" ADD COLUMN "INDUSTRY" TEXT')
    op.execute('ALTER TABLE "ORGANIZATIONS" ADD COLUMN "ADMIN_CONTACT_EMAIL" TEXT')


def downgrade() -> None:
    op.execute('ALTER TABLE "ORGANIZATIONS" DROP COLUMN IF EXISTS "ADMIN_CONTACT_EMAIL"')
    op.execute('ALTER TABLE "ORGANIZATIONS" DROP COLUMN IF EXISTS "INDUSTRY"')
