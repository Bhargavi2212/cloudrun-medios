"""
Patient cache endpoints for orchestrator storage.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from database.session import get_session
from dol_service.schemas.cache import PatientSnapshot, SnapshotAck
from dol_service.schemas.portable_profile import PortableProfileResponse
from dol_service.security.auth import verify_shared_secret
from dol_service.services.patient_cache_service import PatientCacheService

router = APIRouter(prefix="/api/dol/patients", tags=["dol"])


@router.post(
    "/{patient_id}/snapshot",
    response_model=SnapshotAck,
    summary="Ingest patient snapshot",
)
async def ingest_snapshot(
    patient_id: UUID,
    payload: PatientSnapshot,
    requester: str = Depends(verify_shared_secret),
    session: AsyncSession = Depends(get_session),
) -> SnapshotAck:
    """
    Upsert patient demographics and append timeline events.
    """

    if payload.patient.patient_id != patient_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Patient ID mismatch."
        )

    service = PatientCacheService(session)
    ingested = await service.ingest_snapshot(requester, payload)
    await session.commit()
    return SnapshotAck(
        status="cached", patient_id=patient_id, timeline_events_ingested=ingested
    )


@router.get(
    "/search",
    summary="Search for patient by MRN",
    description="Find patient_id (UUID) by MRN for cross-hospital patient matching.",
)
async def search_patient_by_mrn(
    mrn: str = Query(..., description="Medical Record Number to search for"),
    _: str = Depends(verify_shared_secret),
    session: AsyncSession = Depends(get_session),
) -> dict[str, str]:
    """
    Search for a patient in DOL by MRN and return their patient_id (UUID).
    This enables cross-hospital patient matching when the same patient
    has different UUIDs in different hospital systems.
    """
    service = PatientCacheService(session)
    patient_id = await service.find_patient_id_by_mrn(mrn)

    if patient_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient with MRN '{mrn}' not found in DOL.",
        )

    return {"patient_id": str(patient_id), "mrn": mrn}


@router.get(
    "/{patient_id}/profile",
    response_model=PortableProfileResponse,
    summary="Retrieve cached federated profile",
)
async def get_cached_profile(
    patient_id: UUID,
    _: str = Depends(verify_shared_secret),
    session: AsyncSession = Depends(get_session),
) -> PortableProfileResponse:
    """
    Return the cached patient profile and merged timeline.
    """

    service = PatientCacheService(session)
    profile, events = await service.get_cached_profile(patient_id)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Patient not cached."
        )

    patient_model = PatientCacheService.serialize_patient(profile)
    timeline = PatientCacheService.serialize_timeline(events)
    summaries = PatientCacheService.serialize_summaries(profile)
    sources = sorted(
        {profile.primary_hospital_id or patient_model.id}
        | {event.source_hospital_id for event in events}
    )

    return PortableProfileResponse(
        patient=patient_model,
        timeline=timeline,
        summaries=summaries,
        sources=sources,
    )
