"""
Patient routes for the manage-agent service.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from database.session import get_session
from services.manage_agent.schemas import PatientCreate, PatientRead, PatientUpdate
from services.manage_agent.services.patient_service import PatientService

router = APIRouter(prefix="/manage", tags=["patients"])


@router.post(
    "/patients",
    response_model=PatientRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create patient",
    description="Register a new patient within the hospital database.",
)
async def create_patient(
    payload: PatientCreate,
    session: AsyncSession = Depends(get_session),
) -> PatientRead:
    """
    Create a patient record.
    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        from uuid import uuid4

        # Get the raw data and process it
        patient_data = payload.model_dump(exclude_none=False)

        # Generate MRN if not provided or empty
        if not patient_data.get("mrn"):
            patient_data["mrn"] = f"MRN-{uuid4().hex[:8].upper()}"

        # Create a new PatientCreate with processed data
        processed_payload = PatientCreate(**patient_data)

        service = PatientService(session)
        patient = await service.create_patient(processed_payload)
        return PatientRead.model_validate(patient, from_attributes=True)
    except ValueError as e:
        logger.error("Validation error creating patient: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid patient data: {e!s}",
        ) from e
    except Exception as e:
        logger.error("Error creating patient: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create patient: {e!s}",
        ) from e


@router.get(
    "/patients",
    response_model=list[PatientRead],
    summary="List patients",
    description="Retrieve a paginated list of patients.",
)
async def list_patients(
    limit: int = Query(
        default=50, ge=1, le=200, description="Maximum number of records."
    ),
    offset: int = Query(default=0, ge=0, description="Number of records to skip."),
    session: AsyncSession = Depends(get_session),
) -> Sequence[PatientRead]:
    """
    List patients with pagination.
    """

    service = PatientService(session)
    patients = await service.list_patients(limit=limit, offset=offset)
    return [
        PatientRead.model_validate(patient, from_attributes=True)
        for patient in patients
    ]


@router.get(
    "/patients/{patient_id}",
    response_model=PatientRead,
    summary="Retrieve patient",
)
async def get_patient(
    patient_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> PatientRead:
    """
    Fetch a patient by identifier.
    """

    service = PatientService(session)
    patient = await service.get_patient(patient_id)
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found."
        )
    return PatientRead.model_validate(patient, from_attributes=True)


@router.put(
    "/patients/{patient_id}",
    response_model=PatientRead,
    summary="Update patient",
)
async def update_patient(
    patient_id: UUID,
    payload: PatientUpdate,
    session: AsyncSession = Depends(get_session),
) -> PatientRead:
    """
    Update an existing patient record.
    """

    service = PatientService(session)
    patient = await service.get_patient(patient_id)
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found."
        )
    updated = await service.update_patient(patient, payload)
    return PatientRead.model_validate(updated, from_attributes=True)


@router.delete(
    "/patients/{patient_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete patient",
    response_model=None,
)
async def delete_patient(
    patient_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> None:
    """
    Delete a patient record.
    """

    service = PatientService(session)
    patient = await service.get_patient(patient_id)
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found."
        )
    await service.delete_patient(patient)
