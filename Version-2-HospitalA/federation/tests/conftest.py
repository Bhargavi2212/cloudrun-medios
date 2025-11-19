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

from federation.aggregator.main import create_app  # noqa: E402
from federation.config import AggregatorSettings  # noqa: E402


@pytest_asyncio.fixture()
async def client() -> AsyncClient:
    """
    Provide an HTTP client bound to the aggregator FastAPI app.
    """

    settings = AggregatorSettings(
        shared_secret=SHARED_SECRET,
        cors_allow_origins=[],
        debug=True,
    )
    app = create_app(settings)
    headers = {
        "Authorization": f"Bearer {SHARED_SECRET}",
        "X-Hospital-ID": "hospital-a",
    }
    async with AsyncClient(
        app=app, base_url="http://testserver", headers=headers
    ) as test_client:
        yield test_client
