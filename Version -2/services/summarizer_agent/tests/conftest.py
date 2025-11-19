"""
Fixtures for summarizer-agent integration tests.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
if TEST_DATABASE_URL:
    os.environ.setdefault("DATABASE_URL", TEST_DATABASE_URL)

# Imports after environment setup are intentional for test configuration
from database import ensure_loaded
from database.base import Base
from database.models import Patient
from database.session import dispose_engine, init_engine
from services.summarizer_agent.config import SummarizerAgentSettings
from services.summarizer_agent.main import create_app


@pytest_asyncio.fixture(scope="function")
async def prepared_database() -> None:
    """
    Prepare database schema for tests.
    """

    if TEST_DATABASE_URL is None:
        pytest.skip(
            "Set TEST_DATABASE_URL to run summarizer-agent API tests.",
            allow_module_level=True,
        )

    ensure_loaded()
    engine = create_async_engine(TEST_DATABASE_URL, future=True)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def app(prepared_database: None):
    """
    Instantiate the FastAPI application with an initialized engine.
    """

    init_engine(database_url=TEST_DATABASE_URL)
    settings = SummarizerAgentSettings(
        database_url=TEST_DATABASE_URL,  # type: ignore[arg-type]
        cors_allow_origins=[],
        debug=True,
    )
    application = create_app(settings)
    try:
        yield application
    finally:
        await dispose_engine()


@pytest_asyncio.fixture()
async def client(app) -> AsyncClient:
    """
    Provide an AsyncClient bound to the summarizer FastAPI app.
    """

    async with AsyncClient(app=app, base_url="http://testserver") as test_client:
        yield test_client


@pytest_asyncio.fixture()
async def patient_id() -> str:
    """
    Insert a patient record to satisfy foreign key constraints.
    """

    engine = create_async_engine(TEST_DATABASE_URL, future=True)  # type: ignore[arg-type]
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    identifier = uuid4()
    async with session_factory() as session:
        session.add(
            Patient(
                id=identifier,
                mrn=f"SUMMARY-{identifier.hex[:6]}",
                first_name="Casey",
                last_name="Rivera",
            )
        )
        await session.commit()
    await engine.dispose()
    return str(identifier)
