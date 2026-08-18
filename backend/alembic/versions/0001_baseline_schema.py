"""Baseline schema: six tenant-scoped tables.

Identifiers are quoted so Postgres preserves the repo's ALL_CAPS convention
instead of folding them to lowercase.

Revision ID: 0001
"""
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE "GUARDRAIL_CONTACTS" (
            "ID"           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            "TENANT_ID"    TEXT NOT NULL,
            "CUSTOMER_ID"  TEXT NOT NULL,
            "CAMPAIGN_ID"  TEXT NOT NULL,
            "CONTACTED_AT" TIMESTAMPTZ NOT NULL
        )
    """)
    op.execute("""
        CREATE INDEX "IX_GUARDRAIL_CONTACTS_TENANT_CUSTOMER"
            ON "GUARDRAIL_CONTACTS" ("TENANT_ID", "CUSTOMER_ID", "CONTACTED_AT")
    """)
    op.execute("""
        CREATE TABLE "FULFILLMENTS" (
            "TENANT_ID"       TEXT NOT NULL,
            "IDEMPOTENCY_KEY" TEXT NOT NULL,
            "CUSTOMER_ID"     TEXT NOT NULL,
            "CAMPAIGN_ID"     TEXT NOT NULL,
            "ARM"             TEXT NOT NULL,
            "COST"            DOUBLE PRECISION NOT NULL,
            "STATUS"          TEXT NOT NULL,
            "CREATED_AT"      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY ("TENANT_ID", "IDEMPOTENCY_KEY")
        )
    """)
    op.execute("""
        CREATE TABLE "AUDIT_LOG" (
            "ID"          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            "TENANT_ID"   TEXT NOT NULL,
            "NODE"        TEXT NOT NULL,
            "CUSTOMER_ID" TEXT NOT NULL,
            "TS"          TIMESTAMPTZ NOT NULL,
            "PAYLOAD"     JSONB NOT NULL DEFAULT '{}'::jsonb
        )
    """)
    op.execute("""
        CREATE INDEX "IX_AUDIT_LOG_TENANT_CUSTOMER"
            ON "AUDIT_LOG" ("TENANT_ID", "CUSTOMER_ID", "ID" DESC)
    """)
    op.execute("""
        CREATE TABLE "BANDIT_POSTERIOR" (
            "TENANT_ID" TEXT NOT NULL,
            "ARM"       TEXT NOT NULL,
            "A_MATRIX"  BYTEA NOT NULL,
            "B_VECTOR"  BYTEA NOT NULL,
            "N_UPDATES" INTEGER NOT NULL,
            PRIMARY KEY ("TENANT_ID", "ARM")
        )
    """)
    op.execute("""
        CREATE TABLE "MEMORY_EDGES" (
            "ID"          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            "TENANT_ID"   TEXT NOT NULL,
            "CUSTOMER_ID" TEXT NOT NULL,
            "SUBJECT"     TEXT NOT NULL,
            "RELATION"    TEXT NOT NULL,
            "OBJECT"      TEXT NOT NULL,
            "VALID_FROM"  TIMESTAMPTZ NOT NULL,
            "VALID_TO"    TIMESTAMPTZ,
            "EMBEDDING"   BYTEA
        )
    """)
    op.execute("""
        CREATE INDEX "IX_MEMORY_EDGES_TENANT_CUSTOMER"
            ON "MEMORY_EDGES" ("TENANT_ID", "CUSTOMER_ID", "VALID_FROM", "ID")
    """)
    op.execute("""
        CREATE TABLE "SEMANTIC_CACHE" (
            "ID"        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            "TENANT_ID" TEXT NOT NULL,
            "KEY_TEXT"  TEXT NOT NULL,
            "VALUE"     TEXT NOT NULL,
            "EMBEDDING" BYTEA
        )
    """)
    op.execute("""
        CREATE INDEX "IX_SEMANTIC_CACHE_TENANT"
            ON "SEMANTIC_CACHE" ("TENANT_ID")
    """)


def downgrade() -> None:
    for table in (
        "SEMANTIC_CACHE",
        "MEMORY_EDGES",
        "BANDIT_POSTERIOR",
        "AUDIT_LOG",
        "FULFILLMENTS",
        "GUARDRAIL_CONTACTS",
    ):
        op.execute(f'DROP TABLE IF EXISTS "{table}"')
