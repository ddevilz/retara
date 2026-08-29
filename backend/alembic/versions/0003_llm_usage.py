"""Per-tenant LLM token metering.

No COST_USD column: a dollar figure needs a per-model price table that drifts whenever
the provider changes pricing, and a stale price is worse than no price. MODEL plus token
counts plus a timestamp reconstructs cost at any price, whenever billing needs it.

Revision ID: 0003
"""
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE "LLM_USAGE" (
            "ID"         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            "TENANT_ID"  TEXT NOT NULL,
            "ROLE"       TEXT NOT NULL,
            "MODEL"      TEXT NOT NULL,
            "TOKENS_IN"  INTEGER NOT NULL,
            "TOKENS_OUT" INTEGER NOT NULL,
            "TS"         TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    # The budget check aggregates by tenant over a time window; this is the index for it.
    op.execute("""
        CREATE INDEX "IX_LLM_USAGE_TENANT_TS" ON "LLM_USAGE" ("TENANT_ID", "TS")
    """)
    op.execute("""
        ALTER TABLE "ORGANIZATIONS"
        ADD COLUMN "MONTHLY_TOKEN_BUDGET" BIGINT
    """)


def downgrade() -> None:
    op.execute('ALTER TABLE "ORGANIZATIONS" DROP COLUMN IF EXISTS "MONTHLY_TOKEN_BUDGET"')
    op.execute('DROP TABLE IF EXISTS "LLM_USAGE"')
