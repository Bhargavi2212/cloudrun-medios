"""
Fixtures for federation aggregator tests.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest_asyncio
from httpx import AsyncClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SHARED_SECRET = "agg-secret"
os.environ.setdefault("FEDERATION_SHARED_SECRET", SHARED_SECRET)
os.environ.setdefault("FEDERATION_CORS_ORIGINS", '["http://localhost:5173"]')

# Imports after environment setup are intentional for test configuration
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
)

from database.session import (
    dispose_engine,
    get_engine,
    init_engine,
)
from federation.aggregator.main import create_app
from federation.config import AggregatorSettings


@pytest_asyncio.fixture()
async def client() -> AsyncClient:
    """
    Provide an HTTP client bound to the aggregator FastAPI app.
    """

    settings = AggregatorSettings(
        shared_secret=SHARED_SECRET,
        cors_allow_origins=[],
        debug=True,
        database_url=os.getenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:"),
    )
    init_engine(database_url=settings.database_url, echo=settings.debug)
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(_create_sqlite_tables)
    app = create_app(settings)
    headers = {
        "Authorization": f"Bearer {SHARED_SECRET}",
        "X-Hospital-ID": "hospital-a",
    }
    async with AsyncClient(
        app=app, base_url="http://testserver", headers=headers
    ) as test_client:
        yield test_client
    await dispose_engine()


def _create_sqlite_tables(sync_engine) -> None:
    """
    Create a minimal schema for SQLite-based tests (JSONB not supported).
    """

    metadata = MetaData()
    Table(
        "federated_model_rounds",
        metadata,
        Column("id", String(36), primary_key=True),
        Column("model_name", String(128), nullable=False),
        Column("round_id", Integer, nullable=False),
        Column("weights", JSON, nullable=False),
        Column("contributor_count", Integer, nullable=False),
        Column("round_metadata", JSON, nullable=True),
        Column("created_at", DateTime, nullable=True),
        Column("updated_at", DateTime, nullable=True),
    )
    metadata.create_all(sync_engine)
