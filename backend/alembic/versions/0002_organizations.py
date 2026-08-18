"""Tenant registry. ID is the Clerk organization ID verbatim.

Revision ID: 0002
"""
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE "ORGANIZATIONS" (
            "ID"         TEXT PRIMARY KEY,
            "NAME"       TEXT NOT NULL,
            "CREATED_AT" TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    # Money must not live in floating point. Inherited from the original SQLite
    # `COST REAL`, carried into 0001 unchanged. Not a live bug — COST is written
    # once and never summed — but this is the cheapest it will ever be to change,
    # and it gets worse with every phase stacked on the baseline.
    op.execute("""
        ALTER TABLE "FULFILLMENTS"
        ALTER COLUMN "COST" TYPE NUMERIC(12, 2)
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE "FULFILLMENTS"
        ALTER COLUMN "COST" TYPE DOUBLE PRECISION
    """)
    op.execute('DROP TABLE IF EXISTS "ORGANIZATIONS"')
