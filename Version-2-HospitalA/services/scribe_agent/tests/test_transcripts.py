"""
Validation of transcript endpoints.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_create_and_list_transcripts(
    client: AsyncClient, encounter_id: str
) -> None:
    """
    Ensure transcripts can be created and retrieved.
    """

    payload = {
        "encounter_id": encounter_id,
        "transcript": (
            "Doctor: Describe your symptoms.\n" "Patient: I have a sharp headache."
        ),
        "speaker_segments": [
            {"speaker": "doctor", "content": "Describe your symptoms."},
            {"speaker": "patient", "content": "I have a sharp headache."},
        ],
        "source": "scribe",
    }
    create_response = await client.post("/scribe/transcript", json=payload)
    assert create_response.status_code == 201, create_response.text

    list_response = await client.get(f"/scribe/transcript?encounter_id={encounter_id}")
    assert list_response.status_code == 200
    transcripts = list_response.json()
    assert len(transcripts) == 1
    assert transcripts[0]["encounter_id"] == encounter_id
