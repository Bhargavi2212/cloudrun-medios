"""
Patient check-in endpoint that orchestrates DOL profile retrieval.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import Encounter, TriageObservation
from database.session import get_session
from services.manage_agent.dependencies import get_check_in_service
from services.manage_agent.schemas import PortableProfileResponse
from services.manage_agent.services.check_in_service import CheckInService
from services.manage_agent.services.patient_service import PatientService

router = APIRouter(prefix="/manage", tags=["check-in"])


class CheckInRequest(BaseModel):
    """Request payload for patient check-in."""

    patient_id: UUID
    chief_complaint: str
    injury: bool = False
    ambulance_arrival: bool = False
    seen_72h: bool = False


class CheckInResponse(BaseModel):
    """Response from patient check-in."""

    encounter_id: UUID
    triage_level: int | None = None
    profile: dict | None = None
    dol_profile_found: bool | None = None


@router.post(
    "/check-in",
    response_model=CheckInResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Check-in patient with chief complaint",
    description=(
        "Create a new encounter for a patient with chief complaint, "
        "optionally fetch portable profile from DOL service."
    ),
)
async def check_in_patient_with_complaint(
    payload: CheckInRequest,
    session: AsyncSession = Depends(get_session),
    check_in_service: CheckInService = Depends(get_check_in_service),
) -> CheckInResponse:
    """
    Check-in a patient with chief complaint and create an encounter.
    """

    patient_service = PatientService(session)
    patient = await patient_service.get_patient(payload.patient_id)
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found."
        )

    # Create encounter
    encounter = Encounter(
        id=uuid4(),
        patient_id=payload.patient_id,
        arrival_ts=datetime.utcnow(),
    )
    session.add(encounter)
    await session.flush()

    # Create triage observation with chief complaint
    notes = (
        f"Injury: {payload.injury}, Ambulance: {payload.ambulance_arrival}, "
        f"Seen 72h: {payload.seen_72h}"
    )
    triage_obs = TriageObservation(
        id=uuid4(),
        encounter_id=encounter.id,
        chief_complaint=payload.chief_complaint,
        vitals={},
        notes=notes,
    )
    session.add(triage_obs)
    await session.flush()

    # Try to fetch portable profile
    profile_data = None
    dol_profile_found = None
    try:
        profile_response = await check_in_service.fetch_profile(payload.patient_id)
        profile_data = (
            profile_response.model_dump()
            if hasattr(profile_response, "model_dump")
            else None
        )
        dol_profile_found = True
    except Exception:
        # DOL service unavailable or patient not found in network
        dol_profile_found = False

    await session.commit()

    return CheckInResponse(
        encounter_id=encounter.id,
        triage_level=None,  # Triage would be calculated separately
        profile=profile_data,
        dol_profile_found=dol_profile_found,
    )


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
