"""
Patient check-in endpoint that orchestrates DOL profile retrieval.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from database.session import get_session
from services.manage_agent.dependencies import get_check_in_service
from services.manage_agent.schemas import PortableProfileResponse
from services.manage_agent.services.check_in_service import CheckInService
from services.manage_agent.services.patient_service import PatientService

router = APIRouter(prefix="/manage", tags=["check-in"])


@router.post(
    "/patients/{patient_id}/check-in",
    response_model=PortableProfileResponse,
    summary="Check-in patient and retrieve portable profile",
    description=(
        "Automatically fetch the patient's portable profile from the DOL service, "
        "merging remote timelines with local data."
    ),
)
async def check_in_patient(
    patient_id: UUID,
    session: AsyncSession = Depends(get_session),
    check_in_service: CheckInService = Depends(get_check_in_service),
) -> PortableProfileResponse:
    """
    Check-in an existing patient and return their assembled portable profile.
    """

    patient_service = PatientService(session)
    patient = await patient_service.get_patient(patient_id)
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found."
        )

    return await check_in_service.fetch_profile(patient_id)
