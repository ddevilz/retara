"""The reason this project uses a Postgres-backed queue rather than Redis.

With a separate broker these properties are unachievable: the job and the row commit
to different systems, so one can exist without the other.
"""
from sqlalchemy import text

from magenta.auth import ensure_org
from magenta.db import get_engine


def _job_count(conn, tenant_id: str) -> int:
    return conn.execute(
        text(
            "SELECT count(*) FROM procrastinate_jobs "
            "WHERE task_name = 'train_tenant_models' "
            "AND args->>'tenant_id' = :tenant_id"
        ),
        {"tenant_id": tenant_id},
    ).scalar_one()


def test_rollback_leaves_no_orphan_job(migrated_db):
    """The failure mode a Redis broker cannot avoid."""
    engine = get_engine()
    with engine.connect() as conn:
        trans = conn.begin()
        ensure_org(conn, "org_rollback", "Rollback Co")
        assert _job_count(conn, "org_rollback") == 1, "job not visible in-transaction"
        trans.rollback()

    with engine.connect() as conn:
        assert _job_count(conn, "org_rollback") == 0, "orphan job survived rollback"
        exists = conn.execute(
            text('SELECT count(*) FROM "ORGANIZATIONS" WHERE "ID" = :id'),
            {"id": "org_rollback"},
        ).scalar_one()
        assert exists == 0


def test_commit_persists_both_row_and_job(migrated_db):
    engine = get_engine()
    with engine.connect() as conn:
        trans = conn.begin()
        ensure_org(conn, "org_commit", "Commit Co")
        trans.commit()

    with engine.connect() as conn:
        assert _job_count(conn, "org_commit") == 1
        conn.execute(
            text('DELETE FROM "ORGANIZATIONS" WHERE "ID" = :id'), {"id": "org_commit"}
        )
        conn.execute(
            text("DELETE FROM procrastinate_jobs WHERE args->>'tenant_id' = :id"),
            {"id": "org_commit"},
        )
        conn.commit()


def test_existing_org_does_not_requeue(migrated_db):
    """ensure_org runs on EVERY authenticated request. Enqueueing each time would
    queue a training job per request."""
    engine = get_engine()
    with engine.connect() as conn:
        trans = conn.begin()
        ensure_org(conn, "org_repeat", "Repeat Co")
        ensure_org(conn, "org_repeat", "Repeat Co")
        ensure_org(conn, "org_repeat", "Repeat Co")
        assert _job_count(conn, "org_repeat") == 1
        trans.rollback()
