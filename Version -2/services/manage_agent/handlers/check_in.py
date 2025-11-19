"""
Patient check-in endpoint that orchestrates DOL profile retrieval.
"""

from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime
from uuid import UUID, uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import Encounter, TriageObservation
from database.session import get_session
from services.manage_agent.core.triage import TriageEngine
from services.manage_agent.dependencies import (
    get_check_in_service,
    get_federated_sync_service,
    get_profile_merge_service,
    get_triage_engine,
)
from services.manage_agent.schemas import PortableProfileResponse
from services.manage_agent.schemas.queue import CheckInRequest
from services.manage_agent.services.check_in_service import CheckInService
from services.manage_agent.services.federated_sync_service import FederatedSyncService
from services.manage_agent.services.patient_service import PatientService
from services.manage_agent.services.profile_merge_service import ProfileMergeService
from shared.schemas import StandardResponse

router = APIRouter(prefix="/manage", tags=["check-in"])
logger = logging.getLogger(__name__)


@router.post(
    "/check-in",
    response_model=StandardResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Check-in patient with chief complaint",
    description="Create an encounter for a patient with chief complaint and initial triage estimate.",  # noqa: E501
)
async def check_in_patient(
    payload: CheckInRequest,
    session: AsyncSession = Depends(get_session),
    triage_engine: TriageEngine = Depends(get_triage_engine),
    federated_sync: FederatedSyncService | None = Depends(get_federated_sync_service),
    check_in_service: CheckInService | None = Depends(get_check_in_service),
    profile_merge_service: ProfileMergeService = Depends(get_profile_merge_service),
) -> StandardResponse:
    """
    Check-in a patient by creating an encounter with chief complaint.
    """
    print(
        f"[CHECK-IN] Starting check-in for patient_id: {payload.patient_id}, complaint: {payload.chief_complaint}",  # noqa: E501
        file=sys.stderr,
        flush=True,
    )
    logger.info(
        "[CHECK-IN] Starting check-in for patient_id: %s, complaint: %s",
        payload.patient_id,
        payload.chief_complaint,
    )

    # Ensure patient_id is a UUID
    patient_id = payload.patient_id
    if isinstance(patient_id, str):
        try:
            patient_id = UUID(patient_id)
        except ValueError:
            print(
                f"[CHECK-IN] ERROR: Invalid patient_id format: {payload.patient_id}",
                file=sys.stderr,
                flush=True,
            )
            logger.error(
                "[CHECK-IN] ERROR: Invalid patient_id format: %s", payload.patient_id
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid patient_id format.",
            ) from None

    patient_service = PatientService(session)
    patient = await patient_service.get_patient(patient_id)
    if patient is None:
        print(
            f"[CHECK-IN] ERROR: Patient not found: {patient_id}",
            file=sys.stderr,
            flush=True,
        )
        logger.error("[CHECK-IN] ERROR: Patient not found: %s", patient_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found."
        )

    print(
        f"[CHECK-IN] Patient found: {patient.first_name} {patient.last_name} (MRN: {patient.mrn})",  # noqa: E501
        file=sys.stderr,
        flush=True,
    )
    logger.info(
        "[CHECK-IN] Patient found: %s %s (MRN: %s)",
        patient.first_name,
        patient.last_name,
        patient.mrn,
    )

    # Create encounter
    encounter_id = uuid4()
    encounter = Encounter(
        id=encounter_id,
        patient_id=patient_id,
        arrival_ts=datetime.now(UTC),
    )
    session.add(encounter)
    await session.flush()
    print(f"[CHECK-IN] Created encounter: {encounter_id}", file=sys.stderr, flush=True)
    logger.info("[CHECK-IN] Created encounter: %s", encounter_id)

    # Create triage observation with chief complaint
    # Receptionist triage uses only age and chief complaint (no vitals)
    from services.manage_agent.core.receptionist_triage import (
        ReceptionistTriageEngine,
        ReceptionistTriagePayload,
    )

    # Calculate age from DOB if available
    patient_age = None
    if patient.dob:
        from datetime import date

        today = date.today()
        patient_age = (
            today.year
            - patient.dob.year
            - ((today.month, today.day) < (patient.dob.month, patient.dob.day))
        )

    # Use receptionist triage engine (age + chief complaint only)
    receptionist_triage = ReceptionistTriageEngine()
    receptionist_payload = ReceptionistTriagePayload(
        age=patient_age,
        chief_complaint=payload.chief_complaint,
        ambulance_arrival=payload.ambulance_arrival,
        seen_72h=payload.seen_72h,
        injury=payload.injury,
    )

    # Store minimal vitals dict (empty - vitals will be added by nurse)
    vitals_dict = {
        "ambulance_arrival": payload.ambulance_arrival,
        "seen_72h": payload.seen_72h,
        "injury": payload.injury,
    }

    triage_obs = TriageObservation(
        id=uuid4(),
        encounter_id=encounter.id,
        vitals=vitals_dict,
        chief_complaint=payload.chief_complaint,
    )
    session.add(triage_obs)

    # Run receptionist triage classification (age + complaint only)
    try:
        print(
            f"[CHECK-IN] Running receptionist triage (age: {patient_age}, complaint: {payload.chief_complaint})",  # noqa: E501
            file=sys.stderr,
            flush=True,
        )
        logger.info(
            "[CHECK-IN] Running receptionist triage (age: %s, complaint: %s)",
            patient_age,
            payload.chief_complaint,
        )
        triage_result = await receptionist_triage.classify(receptionist_payload)
        triage_obs.triage_score = triage_result.acuity_level
        triage_obs.triage_model_version = triage_result.model_version
        encounter.acuity_level = triage_result.acuity_level
        print(
            f"[CHECK-IN] Triage result: Level {triage_result.acuity_level} (model: {triage_result.model_version})",  # noqa: E501
            file=sys.stderr,
            flush=True,
        )
        logger.info(
            "[CHECK-IN] Triage result: Level %d (model: %s)",
            triage_result.acuity_level,
            triage_result.model_version,
        )
    except Exception as e:
        # If triage fails, default to level 4 (routine)
        print(
            f"[CHECK-IN] WARNING: Triage classification failed, defaulting to level 4: {e}",  # noqa: E501
            file=sys.stderr,
            flush=True,
        )
        logger.warning(
            "[CHECK-IN] WARNING: Triage classification failed, defaulting to level 4: %s",  # noqa: E501
            e,
        )
        triage_obs.triage_score = 4
        encounter.acuity_level = 4

    await session.commit()
    await session.refresh(encounter)
    print(
        f"[CHECK-IN] Check-in completed successfully. Encounter: {encounter.id}, Triage Level: {encounter.acuity_level}",  # noqa: E501
        file=sys.stderr,
        flush=True,
    )
    logger.info(
        "[CHECK-IN] Check-in completed successfully. Encounter: %s, Triage Level: %s",
        encounter.id,
        encounter.acuity_level,
    )

    # Query DOL for cross-hospital profile and merge with local data
    # Use MRN-based matching if patient_id lookup fails
    # (enables cross-hospital patient matching)
    merged_profile = None
    dol_profile_found = False
    if check_in_service is not None:
        try:
            print(
                f"[CHECK-IN] Querying DOL for patient profile: {patient_id} (MRN: {patient.mrn})",  # noqa: E501
                file=sys.stderr,
                flush=True,
            )
            logger.info(
                "[CHECK-IN] Querying DOL for patient profile: %s (MRN: %s)",
                patient_id,
                patient.mrn,
            )
            dol_profile = await check_in_service.fetch_profile(
                patient_id, mrn=patient.mrn
            )
            dol_profile_found = True
            print(
                "[CHECK-IN] DOL profile retrieved, merging with local data",
                file=sys.stderr,
                flush=True,
            )
            logger.info("[CHECK-IN] DOL profile retrieved, merging with local data")
            merged_profile = await profile_merge_service.merge_profiles(
                patient_id, dol_profile
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                print(
                    "[CHECK-IN] Patient not found in DOL, using local profile only",
                    file=sys.stderr,
                    flush=True,
                )
                logger.info(
                    "[CHECK-IN] Patient not found in DOL, using local profile only"
                )
                # Patient not in DOL - use local profile only
                try:
                    merged_profile = await profile_merge_service.merge_profiles(
                        patient_id, None
                    )
                except Exception as merge_error:
                    print(
                        f"[CHECK-IN] WARNING: Failed to build local profile: {merge_error}",  # noqa: E501
                        file=sys.stderr,
                        flush=True,
                    )
                    logger.warning(
                        "[CHECK-IN] WARNING: Failed to build local profile: %s",
                        merge_error,
                    )
            else:
                print(
                    f"[CHECK-IN] WARNING: DOL query failed (status {e.response.status_code}), using local profile only",  # noqa: E501
                    file=sys.stderr,
                    flush=True,
                )
                logger.warning(
                    "[CHECK-IN] WARNING: DOL query failed (status %d), using local profile only",  # noqa: E501
                    e.response.status_code,
                )
                # DOL unavailable - use local profile only
                try:
                    merged_profile = await profile_merge_service.merge_profiles(
                        patient_id, None
                    )
                except Exception as merge_error:
                    print(
                        f"[CHECK-IN] WARNING: Failed to build local profile: {merge_error}",  # noqa: E501
                        file=sys.stderr,
                        flush=True,
                    )
                    logger.warning(
                        "[CHECK-IN] WARNING: Failed to build local profile: %s",
                        merge_error,
                    )
        except Exception as e:
            print(
                f"[CHECK-IN] WARNING: DOL query failed: {e}, using local profile only",
                file=sys.stderr,
                flush=True,
            )
            logger.warning(
                "[CHECK-IN] WARNING: DOL query failed: %s, using local profile only", e
            )
            # DOL unavailable - use local profile only
            try:
                merged_profile = await profile_merge_service.merge_profiles(
                    patient_id, None
                )
            except Exception as merge_error:
                print(
                    f"[CHECK-IN] WARNING: Failed to build local profile: {merge_error}",
                    file=sys.stderr,
                    flush=True,
                )
                logger.warning(
                    "[CHECK-IN] WARNING: Failed to build local profile: %s", merge_error
                )

    # Sync with federated system if available
    if federated_sync is not None:
        try:
            await federated_sync.sync_patient(patient_id)
        except Exception:
            pass  # Don't fail check-in if federation sync fails

    response_data = {
        "encounter_id": str(encounter.id),
        "triage_level": encounter.acuity_level,
    }

    # Include merged profile if available
    if merged_profile is not None:
        response_data["profile"] = merged_profile.model_dump()
        response_data["dol_profile_found"] = dol_profile_found

    return StandardResponse(
        success=True,
        data=response_data,
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
async def check_in_patient_legacy(
    patient_id: UUID,
    session: AsyncSession = Depends(get_session),
    check_in_service: CheckInService = Depends(get_check_in_service),
    federated_sync: FederatedSyncService | None = Depends(get_federated_sync_service),
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

    profile = await check_in_service.fetch_profile(patient_id)
    if federated_sync is not None:
        await federated_sync.sync_patient(patient_id)
    return profile
