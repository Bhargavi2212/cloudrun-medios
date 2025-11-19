"""
Summarizer-agent endpoint tests.
"""

from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_generate_and_fetch_summary(client: AsyncClient, patient_id: str) -> None:
    """
    Ensure summary generation stores and retrieves data correctly.
    """

    encounter_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
    payload = {
        "patient_id": patient_id,
        "encounter_ids": encounter_ids,
        "highlights": [
            "Encounter 1: Treated for migraine.",
            "Encounter 2: Follow-up visit with improved symptoms.",
        ],
    }
    response = await client.post("/summarizer/generate-summary", json=payload)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["patient_id"] == patient_id
    assert body["encounter_ids"] == encounter_ids

    history_response = await client.get(f"/summarizer/history/{patient_id}")
    assert history_response.status_code == 200
    history = history_response.json()
    assert len(history) >= 1
    assert history[0]["summary_text"].startswith(f"Patient {patient_id}")
