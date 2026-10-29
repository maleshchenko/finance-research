"""Alembic environment — async-aware migration runner."""

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.database.models import Base

# ── Resolve the database URL ──────────────────────────────────────────────────
# Read straight from the environment; fall back to the docker-compose default.
_async_url: str = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://user:password@localhost:5432/financial_research",
)
# Alembic's offline mode needs a synchronous driver
_sync_url: str = _async_url.replace("+asyncpg", "+psycopg2")

# ── Alembic config ────────────────────────────────────────────────────────────
config = context.config
config.set_main_option("sqlalchemy.url", _sync_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


# ── Migration runners ─────────────────────────────────────────────────────────

def run_migrations_offline() -> None:
    context.configure(
        url=_sync_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    async_engine = async_engine_from_config(
        {"sqlalchemy.url": _async_url},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with async_engine.connect() as conn:
        await conn.run_sync(do_run_migrations)
    await async_engine.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
