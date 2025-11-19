"""
Tests for DOL federated endpoints.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

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


async def test_registry_upsert_and_list(client: AsyncClient) -> None:
    """
    Ensure hospitals can register and be listed.
    """

    payload = {
        "hospital_id": "hospital-b",
        "name": "County Hospital B",
        "manage_url": "http://localhost:9001",
        "scribe_url": "http://localhost:9002",
        "summarizer_url": "http://localhost:9003",
        "dol_url": "http://localhost:9004",
        "capabilities": ["triage", "scribe"],
    }
    response = await client.post("/api/dol/registry", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == payload["hospital_id"]
    assert body["status"] == "online"

    listing = await client.get("/api/dol/registry")
    assert listing.status_code == 200
    entries = listing.json()
    assert any(entry["id"] == payload["hospital_id"] for entry in entries)


async def test_patient_snapshot_cache_and_fetch(
    client: AsyncClient, patient_id: str
) -> None:
    """
    Validate patient snapshot ingestion and retrieval.
    """

    summary_id = str(uuid4())
    timeline_event_id = str(uuid4())
    snapshot = {
        "patient": {
            "patient_id": patient_id,
            "mrn": "CACHE-001",
            "first_name": "Jamie",
            "last_name": "Lopez",
        },
        "summaries": [
            {
                "id": summary_id,
                "encounter_ids": [],
                "summary_text": "Patient stable.",
                "model_version": "summary_v1",
                "confidence_score": 0.92,
                "created_at": datetime.now(tz=UTC).isoformat(),
            }
        ],
        "timeline": [
            {
                "external_id": timeline_event_id,
                "event_type": "encounter_note",
                "event_timestamp": datetime.now(tz=UTC).isoformat(),
                "content": {"notes": "Follow-up completed."},
            }
        ],
    }
    ingest = await client.post(
        f"/api/dol/patients/{patient_id}/snapshot", json=snapshot
    )
    assert ingest.status_code == 200, ingest.text
    ack = ingest.json()
    assert ack["timeline_events_ingested"] == 1

    profile_response = await client.get(f"/api/dol/patients/{patient_id}/profile")
    assert profile_response.status_code == 200, profile_response.text
    payload = profile_response.json()
    assert payload["patient"]["id"] == patient_id
    assert len(payload["timeline"]) >= 1
    assert payload["summaries"][0]["id"] == summary_id
