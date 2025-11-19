"""
SOAP generation endpoint tests.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_generate_soap_note(client: AsyncClient, encounter_id: str) -> None:
    """
    Verify SOAP note generation and persistence.
    """

    payload = {
        "encounter_id": encounter_id,
        "transcript": "Patient complains of persistent cough lasting three days.",
    }
    response = await client.post("/scribe/generate-soap", json=payload)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["encounter_id"] == encounter_id
    assert body["assessment"].lower().startswith("provisional")
