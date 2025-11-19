"""
Business logic for patient management.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from database.crud import AsyncCRUDRepository
from database.models import Patient
from services.manage_agent.schemas.patient import PatientCreate, PatientUpdate


class PatientService:
    """
    Coordinates patient persistence concerns.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repository = AsyncCRUDRepository[Patient](session, Patient)

    async def create_patient(self, payload: PatientCreate) -> Patient:
        """
        Create and persist a patient record.
        """

        patient = await self.repository.create(payload.model_dump(exclude_none=True))
        await self.session.commit()
        await self.session.refresh(patient)
        return patient

    async def list_patients(
        self, *, limit: int = 100, offset: int = 0
    ) -> Sequence[Patient]:
        """
        Return a paginated list of patients.
        """

        return await self.repository.list(limit=limit, offset=offset)

    async def get_patient(self, patient_id: UUID) -> Patient | None:
        """
        Retrieve a patient by identifier.
        """

        return await self.repository.get(patient_id)

    async def update_patient(self, patient: Patient, payload: PatientUpdate) -> Patient:
        """
        Apply updates to a patient record.
        """

        updated = await self.repository.update(
            patient, payload.model_dump(exclude_none=True)
        )
        await self.session.commit()
        await self.session.refresh(updated)
        return updated

    async def delete_patient(self, patient: Patient) -> None:
        """
        Delete a patient record.
        """

        await self.repository.delete(patient)
        await self.session.commit()
