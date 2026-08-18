"""Alembic environment. Reads DATABASE_URL through magenta.db so there is exactly
one source of truth for the connection string."""
from alembic import context
from sqlalchemy import engine_from_config, pool

from magenta.db import database_url

config = context.config
config.set_main_option("sqlalchemy.url", database_url())
target_metadata = None  # deliberate SQL, no ORM metadata to autogenerate from


def run_migrations_offline() -> None:
    context.configure(url=database_url(), literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
