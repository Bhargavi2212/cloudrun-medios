"""
Fixtures for scribe-agent integration tests.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
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
from database.models import Encounter, Patient
from database.session import dispose_engine, init_engine
from services.scribe_agent.config import ScribeAgentSettings
from services.scribe_agent.main import create_app


@pytest_asyncio.fixture(scope="function")
async def prepared_database() -> None:
    """
    Prepare schema for the test database.
    """

    if TEST_DATABASE_URL is None:
        pytest.skip(
            "Set TEST_DATABASE_URL to run scribe-agent API tests.",
            allow_module_level=True,
        )

    ensure_loaded()
    engine = create_async_engine(TEST_DATABASE_URL, future=True)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)
    await engine.dispose()


@pytest_asyncio.fixture()
async def app(prepared_database: None):
    """
    Instantiate the FastAPI application with initialized engine.
    """

    init_engine(database_url=TEST_DATABASE_URL)
    settings = ScribeAgentSettings(
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
    Provide an HTTP client for API tests.
    """

    async with AsyncClient(app=app, base_url="http://testserver") as test_client:
        yield test_client


@pytest_asyncio.fixture()
async def encounter_id() -> str:
    """
    Insert patient and encounter prerequisites.
    """

    engine = create_async_engine(TEST_DATABASE_URL, future=True)  # type: ignore[arg-type]
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    patient_identifier = uuid4()
    encounter_identifier = uuid4()

    async with session_factory() as session:
        session.add(
            Patient(
                id=patient_identifier,
                mrn=f"SCRIBE-{patient_identifier.hex[:6]}",
                first_name="Robin",
                last_name="Chen",
            )
        )
        session.add(
            Encounter(
                id=encounter_identifier,
                patient_id=patient_identifier,
                arrival_ts=datetime.now(tz=UTC),
            )
        )
        await session.commit()

    await engine.dispose()
    return str(encounter_identifier)
