"""
Tests for triage classification endpoint.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_triage_classification_returns_acuity(client: AsyncClient) -> None:
    """
    Ensure triage classification returns expected schema.
    """

    payload = {
        "hr": 110,
        "rr": 22,
        "sbp": 118,
        "dbp": 70,
        "temp_c": 37.5,
        "spo2": 95,
        "pain": 3,
    }
    response = await client.post("/manage/classify", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert {"acuity_level", "model_version", "explanation"} <= body.keys()
    assert 1 <= body["acuity_level"] <= 5
