"""
Async SQLAlchemy session and engine management.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable
from typing import Any

from pydantic import Field, ValidationError
from pydantic_settings import SettingsConfigDict
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from shared.config import BaseAppSettings, get_settings
from shared.exceptions import ConfigurationError

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


class DatabaseSettings(BaseAppSettings):
    """
    Application settings related to the database connection.
    """

    database_url: str = Field(alias="DATABASE_URL")
    echo: bool = Field(default=False, alias="DATABASE_ECHO")
    pool_size: int = Field(default=10, alias="DATABASE_POOL_SIZE")
    max_overflow: int = Field(default=20, alias="DATABASE_MAX_OVERFLOW")
    model_config = SettingsConfigDict(
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


def init_engine(**overrides: Any) -> None:
    """
    Initialize the global async engine and session factory.

    Args:
        overrides: Optional keyword arguments overriding settings fields.

    Raises:
        ConfigurationError: If no database URL is supplied.
    """

    settings = _resolve_settings(overrides)
    database_url = settings.database_url

    # DEBUG & FIX: Ensure async driver is used
    import logging

    logger = logging.getLogger(__name__)

    if database_url.startswith("postgresql://") and "asyncpg" not in database_url:
        scheme = database_url.split("://")[0]
        logger.warning(
            f"Database URL scheme '{scheme}' might default to psycopg2 (sync)."
        )
        logger.warning("Attempting to auto-switch to 'postgresql+asyncpg://'...")
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")

    logger.info(
        f"Initializing database engine with URL scheme: {database_url.split('://')[0]}"
    )

    global _engine  # pylint: disable=global-statement
    global _session_factory  # pylint: disable=global-statement

    engine_kwargs: dict[str, Any] = {
        "echo": settings.echo,
        "pool_pre_ping": True,
    }
    if database_url.startswith("sqlite"):
        engine_kwargs["poolclass"] = StaticPool
    else:
        engine_kwargs["pool_size"] = settings.pool_size
        engine_kwargs["max_overflow"] = settings.max_overflow

    _engine = create_async_engine(
        database_url,
        **engine_kwargs,
    )
    _session_factory = async_sessionmaker(
        bind=_engine,
        expire_on_commit=False,
        autoflush=False,
    )


def _resolve_settings(overrides: dict[str, Any]) -> DatabaseSettings:
    """
    Resolve database settings by merging environment values and overrides.
    """

    if overrides.get("database_url"):
        return DatabaseSettings(**overrides)

    try:
        return get_settings(DatabaseSettings)
    except ValidationError as exc:  # pragma: no cover - validation path
        raise ConfigurationError(
            "DATABASE_URL environment variable is required."
        ) from exc


def get_engine() -> AsyncEngine:
    """
    Return the initialized async engine.

    Raises:
        ConfigurationError: If the engine has not been initialized.
    """

    if _engine is None:
        raise ConfigurationError(
            "Database engine has not been initialized. Call init_engine()."
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """
    Return the initialized async session factory.

    Raises:
        ConfigurationError: If the session factory has not been initialized.
    """

    if _session_factory is None:
        raise ConfigurationError(
            "Session factory is not initialized. Call init_engine()."
        )
    return _session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields an AsyncSession.
    """

    session_factory = get_session_factory()
    async with session_factory() as session:
        yield session


async def dispose_engine() -> None:
    """
    Dispose of the underlying engine.
    """

    if _engine is not None:
        await _engine.dispose()


def run_sync(coro: Awaitable[Any]) -> Any:
    """
    Run an async callable synchronously (useful for scripts).

    Args:
        coro: Awaitable object to run.
    """

    return asyncio.run(coro)
