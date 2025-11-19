"""
Fixtures for DOL service tests.
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_DATABASE_URL = (
    os.environ.get("TEST_DATABASE_URL")
    or os.getenv("TEST_DATABASE_URL", "postgresql+asyncpg://postgres:test_password@localhost:5432/medi_os_v2_b_test")
)
os.environ["TEST_DATABASE_URL"] = TEST_DATABASE_URL
os.environ["DATABASE_URL"] = TEST_DATABASE_URL
os.environ["DOL_HOSPITAL_ID"] = "hospital-a"
os.environ["DOL_SHARED_SECRET"] = "test-shared-secret"
os.environ["DOL_CORS_ORIGINS"] = '["http://localhost:5173"]'
os.environ["DOL_PEERS"] = "[]"
SHARED_SECRET = "test-shared-secret"

from database import ensure_loaded
from database.base import Base
from database.models import (
    DialogueTranscript,
    Encounter,
    Patient,
    SoapNote,
    Summary,
    TriageObservation,
)
from database.session import dispose_engine, init_engine
from dol_service.config import DOLSettings
from dol_service.main import create_app


@pytest_asyncio.fixture(scope="function")
async def prepared_database() -> None:
    """
    Initialize the test database schema.
    """

    if TEST_DATABASE_URL is None:
        pytest.skip(
            "Set TEST_DATABASE_URL to run DOL API tests.", allow_module_level=True
        )

    ensure_loaded()
    engine = create_async_engine(TEST_DATABASE_URL, future=True)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)
    await engine.dispose()


@pytest_asyncio.fixture()
async def session_factory() -> async_sessionmaker:
    """
    Provide a session factory bound to the test database.
    """

    engine = create_async_engine(TEST_DATABASE_URL, future=True)  # type: ignore[arg-type]
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        await engine.dispose()


@pytest_asyncio.fixture()
async def patient_id(session_factory: async_sessionmaker) -> str:
    """
    Seed a patient with related records for testing.
    """

    patient_identifier = uuid4()
    encounter_identifier = uuid4()
    async with session_factory() as session:
        session.add(
            Patient(
                id=patient_identifier,
                mrn="FED-001",
                first_name="Jamie",
                last_name="Lopez",
            )
        )
        session.add(
            Encounter(
                id=encounter_identifier,
                patient_id=patient_identifier,
                arrival_ts=datetime.now(tz=UTC),
                disposition="discharged",
                acuity_level=3,
            )
        )
        session.add(
            TriageObservation(
                id=uuid4(),
                encounter_id=encounter_identifier,
                vitals={"hr": 90, "rr": 18},
                triage_score=3,
            )
        )
        session.add(
            DialogueTranscript(
                id=uuid4(),
                encounter_id=encounter_identifier,
                transcript="Doctor: Describe symptoms.\nPatient: Mild chest pain.",
            )
        )
        session.add(
            SoapNote(
                id=uuid4(),
                encounter_id=encounter_identifier,
                subjective="Patient reports mild chest pain.",
                assessment="Atypical chest pain.",
                plan="Discharge with follow-up.",
            )
        )
        session.add(
            Summary(
                id=uuid4(),
                patient_id=patient_identifier,
                encounter_ids=[str(encounter_identifier)],
                summary_text="Chest pain evaluated and discharged.",
                model_version="summary_v0",
                confidence_score=0.7,
            )
        )
        await session.commit()
    return str(patient_identifier)


@pytest_asyncio.fixture(scope="function")
async def app(prepared_database: None):
    """
    Instantiate the DOL service application with initialized engine.
    """

    init_engine(database_url=TEST_DATABASE_URL)
    settings = DOLSettings(
        database_url=TEST_DATABASE_URL,  # type: ignore[arg-type]
        hospital_id="hospital-a",
        shared_secret=SHARED_SECRET,
        cors_allow_origins=[],
        peers=[],
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
    Provide a test client for the DOL API.
    """

    headers = {"Authorization": f"Bearer {SHARED_SECRET}", "X-Requester": "test-peer"}
    async with AsyncClient(
        app=app, base_url="http://testserver", headers=headers
    ) as test_client:
        yield test_client
