"""
Federated data orchestration endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from database.session import get_session
from dol_service.core.privacy_filter import redact_metadata, sanitize_timeline
from dol_service.dependencies import get_hospital_id, get_peer_client
from dol_service.schemas.model_update import ModelUpdateRequest, ModelUpdateResponse
from dol_service.schemas.portable_profile import (
    FederatedPatientRequest,
    FederatedTimelineResponse,
    PortablePatient,
    PortableProfileResponse,
    PortableSummary,
    PortableTimelineEvent,
)
from dol_service.security.auth import verify_shared_secret
from dol_service.services.audit_service import AuditService
from dol_service.services.peer_client import PeerClient
from dol_service.services.profile_merger import merge_profiles
from dol_service.services.profile_service import ProfileService

router = APIRouter(prefix="/api/federated", tags=["federated"])


@router.post(
    "/patient",
    response_model=PortableProfileResponse,
    summary="Retrieve portable patient profile",
)
async def get_portable_profile(
    payload: FederatedPatientRequest,
    requester: str = Depends(verify_shared_secret),
    session: AsyncSession = Depends(get_session),
    peer_client: PeerClient = Depends(get_peer_client),
    hospital_id: str = Depends(get_hospital_id),
) -> PortableProfileResponse:
    """
    Assemble a privacy-filtered patient profile across local and peer hospitals.
    """

    profile_service = ProfileService(session, hospital_id)
    local_profile = await profile_service.build_profile(payload.patient_id)
    if local_profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found."
        )

    remote_profiles = await peer_client.fetch_profiles(payload.patient_id)
    merged = merge_profiles(local_profile, remote_profiles)

    sanitized_patient = redact_metadata(merged["patient"])
    sanitized_timeline = sanitize_timeline(merged["timeline"])
    sanitized_summaries = [redact_metadata(summary) for summary in merged["summaries"]]

    timeline_models = [
        PortableTimelineEvent.model_validate(event) for event in sanitized_timeline
    ]
    summary_models = [
        PortableSummary.model_validate(summary) for summary in sanitized_summaries
    ]
    patient_model = PortablePatient.model_validate(sanitized_patient)

    audit_service = AuditService(session)
    await audit_service.log_access(
        patient_id=payload.patient_id,
        requester=requester,
        action="federated_profile_read",
    )

    return PortableProfileResponse(
        patient=patient_model,
        timeline=timeline_models,
        summaries=summary_models,
        sources=merged["sources"],
    )


@router.post(
    "/timeline",
    response_model=FederatedTimelineResponse,
    summary="Retrieve local timeline fragment",
)
async def get_timeline_fragment(
    payload: FederatedPatientRequest,
    requester: str = Depends(verify_shared_secret),
    session: AsyncSession = Depends(get_session),
    hospital_id: str = Depends(get_hospital_id),
) -> FederatedTimelineResponse:
    """
    Provide a timeline fragment to a peer hospital.
    """

    profile_service = ProfileService(session, hospital_id)
    local_profile = await profile_service.build_profile(payload.patient_id)
    if local_profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found."
        )

    sanitized_timeline = sanitize_timeline(local_profile["timeline"])
    sanitized_summaries = [
        redact_metadata(summary) for summary in local_profile["summaries"]
    ]

    timeline_models = [
        PortableTimelineEvent.model_validate(event) for event in sanitized_timeline
    ]
    summary_models = [
        PortableSummary.model_validate(summary) for summary in sanitized_summaries
    ]

    audit_service = AuditService(session)
    await audit_service.log_access(
        patient_id=payload.patient_id,
        requester=requester,
        action="federated_timeline_read",
    )

    return FederatedTimelineResponse(
        timeline=timeline_models,
        summaries=summary_models,
        source_hospital=hospital_id,
    )


@router.post(
    "/model_update",
    response_model=ModelUpdateResponse,
    summary="Receive model update payload",
)
async def receive_model_update(
    payload: ModelUpdateRequest,
    requester: str = Depends(verify_shared_secret),
) -> ModelUpdateResponse:
    """
    Accept a model update from a peer (pass-through acknowledgement for now).
    """

    return ModelUpdateResponse(
        status=f"queued by {requester}",
        model_name=payload.model_name,
        round_id=payload.round_id,
    )
