"""
Shared pytest fixtures for database tests.
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from database import ensure_loaded
from database.base import Base

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")


@pytest_asyncio.fixture()
async def engine() -> AsyncEngine:
    """
    Provide a database engine for tests. Requires TEST_DATABASE_URL.
    """

    if TEST_DATABASE_URL is None:
        pytest.skip(
            "Set TEST_DATABASE_URL to run database integration tests.",
            allow_module_level=True,
        )

    ensure_loaded()
    engine = create_async_engine(TEST_DATABASE_URL, future=True)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture()
async def db_session(engine: AsyncEngine) -> AsyncSession:
    """
    Yield a database session with rollback on completion.
    """

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        try:
            yield session
        finally:
            await session.rollback()
