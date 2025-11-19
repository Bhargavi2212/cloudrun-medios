"""
Aggregator endpoint tests.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_submit_and_fetch_model(client: AsyncClient) -> None:
    """
    Submit two updates and verify aggregated result.
    """

    update_payload = {
        "model_name": "triage",
        "round_id": 1,
        "hospital_id": "hospital-a",
        "weights": {"layer1": [0.2, 0.4], "layer2": [0.1]},
    }
    response = await client.post("/federation/submit", json=update_payload)
    assert response.status_code == 200, response.text
    ack = response.json()
    assert ack["contributor_count"] == 1

    # second update from another hospital
    client.headers["X-Hospital-ID"] = "hospital-b"
    update_payload["hospital_id"] = "hospital-b"
    update_payload["weights"] = {"layer1": [0.4, 0.6], "layer2": [0.3]}
    response = await client.post("/federation/submit", json=update_payload)
    assert response.status_code == 200
    ack = response.json()
    assert ack["contributor_count"] == 2

    global_response = await client.get("/federation/global-model/triage")
    assert global_response.status_code == 200
    model = global_response.json()
    assert model["weights"]["layer1"] == pytest.approx([0.3, 0.5])
    assert model["contributor_count"] == 2
