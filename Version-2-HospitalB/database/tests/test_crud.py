"""
Tests for the generic CRUD repository.
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from database.crud import AsyncCRUDRepository
from database.models import Patient

pytestmark = pytest.mark.asyncio


async def test_async_crud_repository_lifecycle(db_session: AsyncSession) -> None:
    """
    Verify create, list, update, and delete behaviors.
    """

    repository = AsyncCRUDRepository[Patient](db_session, Patient)
    patient = await repository.create(
        {
            "mrn": "TEST-CRUD-01",
            "first_name": "Jordan",
            "last_name": "Nguyen",
            "dob": date(1985, 2, 17),
            "sex": "male",
            "contact_info": {"phone": "+1-555-0001"},
        }
    )

    patients = await repository.list()
    assert any(row.id == patient.id for row in patients)

    await repository.update(
        patient,
        {"contact_info": {"phone": "+1-555-9999", "email": "jordan@example.com"}},
    )
    refreshed = await repository.get(patient.id)
    assert refreshed is not None
    assert refreshed.contact_info["email"] == "jordan@example.com"

    await repository.delete(refreshed)
    assert await repository.get(patient.id) is None
