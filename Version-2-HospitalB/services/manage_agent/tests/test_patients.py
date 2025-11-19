"""
API tests for patient endpoints.
"""

from __future__ import annotations

from uuid import UUID

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_patient_lifecycle(client: AsyncClient) -> None:
    """
    Validate create, read, update, delete operations.
    """

    payload = {
        "mrn": "API-TEST-001",
        "first_name": "Taylor",
        "last_name": "Brooks",
        "sex": "nonbinary",
        "contact_info": {"email": "taylor.brooks@example.com"},
    }
    create_response = await client.post("/manage/patients", json=payload)
    assert create_response.status_code == 201, create_response.text

    body = create_response.json()
    patient_id = UUID(body["id"])
    assert body["mrn"] == payload["mrn"]

    detail_response = await client.get(f"/manage/patients/{patient_id}")
    assert detail_response.status_code == 200
    assert detail_response.json()["first_name"] == "Taylor"

    update_response = await client.put(
        f"/manage/patients/{patient_id}",
        json={"contact_info": {"email": "updated@example.com"}},
    )
    assert update_response.status_code == 200
    assert update_response.json()["contact_info"]["email"] == "updated@example.com"

    list_response = await client.get("/manage/patients")
    assert list_response.status_code == 200
    assert any(row["id"] == str(patient_id) for row in list_response.json())

    delete_response = await client.delete(f"/manage/patients/{patient_id}")
    assert delete_response.status_code == 204

    missing_response = await client.get(f"/manage/patients/{patient_id}")
    assert missing_response.status_code == 404
