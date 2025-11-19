"""
Tests for the patient check-in endpoint.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient

from services.manage_agent.dependencies import get_check_in_service
from services.manage_agent.schemas.portable_profile import (
    PortablePatient,
    PortableProfileResponse,
    PortableSummary,
    PortableTimelineEvent,
)

pytestmark = pytest.mark.asyncio


class StubCheckInService:
    """
    Stub service returning deterministic portable profiles.
    """

    async def fetch_profile(
        self, patient_id: UUID
    ) -> PortableProfileResponse:  # pragma: no cover - trivial
        return PortableProfileResponse(
            patient=PortablePatient(
                id=str(patient_id),
                mrn="MED-999",
                first_name="Jamie",
                last_name="Lopez",
                dob=None,
                sex=None,
                contact_info=None,
            ),
            timeline=[
                PortableTimelineEvent(
                    event_type="encounter",
                    encounter_id="enc-1",
                    timestamp="2023-11-15T00:00:00+00:00",
                    content={"disposition": "discharged"},
                )
            ],
            summaries=[
                PortableSummary(
                    id="sum-1",
                    encounter_ids=["enc-1"],
                    summary_text="Patient discharged with inhaler.",
                    model_version="summary_v0",
                    confidence_score=0.8,
                    created_at="2023-11-15T00:10:00+00:00",
                )
            ],
            sources=["hospital-a", "hospital-b"],
        )


async def test_check_in_returns_portable_profile(app, client: AsyncClient) -> None:
    """
    Verify the check-in endpoint proxies portable profile data.
    """

    create_response = await client.post(
        "/manage/patients",
        json={
            "mrn": "TEST-222",
            "first_name": "Casey",
            "last_name": "Rivera",
        },
    )
    assert create_response.status_code == 201, create_response.text
    patient_id = create_response.json()["id"]

    stub = StubCheckInService()
    app.dependency_overrides[get_check_in_service] = lambda: stub

    try:
        response = await client.post(f"/manage/patients/{patient_id}/check-in")
    finally:
        app.dependency_overrides.pop(get_check_in_service, None)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["patient"]["mrn"] == "MED-999"
    assert body["sources"] == ["hospital-a", "hospital-b"]


async def test_check_in_patient_not_found(app, client: AsyncClient) -> None:
    """
    Ensure a 404 is returned when the patient does not exist locally.
    """

    response = await client.post(f"/manage/patients/{uuid4()}/check-in")
    assert response.status_code == 404
