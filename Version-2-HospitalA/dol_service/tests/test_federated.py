"""
Tests for DOL federated endpoints.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_get_portable_profile(client: AsyncClient, patient_id: str) -> None:
    """
    Ensure the portable profile endpoint returns sanitized data.
    """

    response = await client.post(
        "/api/federated/patient", json={"patient_id": patient_id}
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["patient"]["id"] == patient_id
    assert "hospital_id" not in body["patient"]
    assert body["sources"] == ["hospital-a"]
    assert all("hospital_id" not in event for event in body["timeline"])


async def test_get_timeline_fragment(client: AsyncClient, patient_id: str) -> None:
    """
    Validate the timeline fragment endpoint for peer consumption.
    """

    response = await client.post(
        "/api/federated/timeline", json={"patient_id": patient_id}
    )
    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["source_hospital"] == "hospital-a"
    assert len(payload["timeline"]) >= 1
    assert "hospital_id" not in payload["timeline"][0]


async def test_model_update_acknowledgement(client: AsyncClient) -> None:
    """
    Ensure model updates receive acknowledgement.
    """

    response = await client.post(
        "/api/federated/model_update",
        json={
            "model_name": "triage",
            "round_id": 1,
            "payload": {"weights": [0.1, 0.2]},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"].startswith("queued by")
