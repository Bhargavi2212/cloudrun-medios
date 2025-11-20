"""
Test fixtures for the manage-agent service.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
if TEST_DATABASE_URL:
    os.environ.setdefault("DATABASE_URL", TEST_DATABASE_URL)

# Imports after environment setup are intentional for test configuration
from database import ensure_loaded
from database.base import Base
from database.session import dispose_engine, init_engine
from services.manage_agent.config import ManageAgentSettings
from services.manage_agent.main import create_app


@pytest_asyncio.fixture(scope="function")
async def prepared_database() -> None:
    """
    Ensure the test database schema exists.
    """

    if TEST_DATABASE_URL is None:
        pytest.skip(
            "Set TEST_DATABASE_URL to run manage-agent API tests.",
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
    settings = ManageAgentSettings(
        database_url=TEST_DATABASE_URL,  # type: ignore[arg-type]
        cors_allow_origins=["*"],  # Allow all origins in tests
        debug=True,
        dol_base_url="http://dol-service",
        dol_shared_secret="test-secret",
        dol_hospital_id="hospital-a",
    )
    application = create_app(settings)
    try:
        yield application
    finally:
        await dispose_engine()


@pytest_asyncio.fixture()
async def client(app) -> AsyncClient:
    """
    Provide an HTTP client bound to the FastAPI application.
    """

    async with AsyncClient(app=app, base_url="http://testserver") as test_client:
        yield test_client
