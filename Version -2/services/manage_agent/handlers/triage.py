"""
Triage classification endpoints.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from database.models import Encounter, TriageObservation
from database.session import get_session
from services.manage_agent.core.nurse_triage import (
    NurseTriageEngine,
    NurseTriagePayload,
)
from services.manage_agent.core.triage import TriageEngine
from services.manage_agent.dependencies import (
    get_nurse_triage_engine,
    get_triage_engine,
)
from services.manage_agent.schemas import (
    NurseVitalsRequest,
    NurseVitalsResponse,
    TriageRequest,
    TriageResponse,
)
from shared.schemas import StandardResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/manage", tags=["triage"])


@router.post(
    "/classify",
    response_model=TriageResponse,
    summary="Classify triage acuity",
    description="Generate an acuity score based on vital signs.",
)
async def classify_patient(
    payload: TriageRequest,
    engine: TriageEngine = Depends(get_triage_engine),
) -> TriageResponse:
    """
    Run the triage engine against incoming vital signs.
    """

    result = await engine.classify(payload)
    return TriageResponse.model_validate(asdict(result))


@router.post(
    "/encounters/{encounter_id}/vitals",
    response_model=StandardResponse,
    summary="Record vitals and update triage",
    description="Capture vitals for an encounter and run the nurse triage model with vitals.",  # noqa: E501
)
async def record_vitals(
    encounter_id: UUID,
    payload: NurseVitalsRequest,
    session: AsyncSession = Depends(get_session),
    engine: NurseTriageEngine = Depends(get_nurse_triage_engine),
) -> StandardResponse:
    """
    Nurse workflow: store vitals and classify acuity for an encounter.
    """
    import sys

    print(
        f"[VITALS] Recording vitals for encounter: {encounter_id}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[VITALS] Vitals: HR={payload.hr}, RR={payload.rr}, BP={payload.sbp}/{payload.dbp}, Temp={payload.temp_c}C, SpO2={payload.spo2}, Pain={payload.pain}",  # noqa: E501
        file=sys.stderr,
        flush=True,
    )
    logger.info("[VITALS] Recording vitals for encounter: %s", encounter_id)
    logger.info(
        "[VITALS] Vitals: HR=%d, RR=%d, BP=%d/%d, Temp=%.1fC, SpO2=%d, Pain=%d",
        payload.hr,
        payload.rr,
        payload.sbp,
        payload.dbp,
        payload.temp_c,
        payload.spo2,
        payload.pain,
    )

    stmt = (
        select(Encounter)
        .where(Encounter.id == encounter_id)
        .options(
            selectinload(Encounter.triage_observations),
            selectinload(Encounter.patient),
        )
    )
    result = await session.execute(stmt)
    encounter = result.scalar_one_or_none()
    if encounter is None:
        print(
            f"[VITALS] ERROR: Encounter not found: {encounter_id}",
            file=sys.stderr,
            flush=True,
        )
        logger.error("[VITALS] ERROR: Encounter not found: %s", encounter_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Encounter not found.",
        )

    print(
        f"[VITALS] Encounter found for patient: {encounter.patient_id if encounter.patient else 'unknown'}",  # noqa: E501
        file=sys.stderr,
        flush=True,
    )
    logger.info(
        "[VITALS] Encounter found for patient: %s",
        encounter.patient_id if encounter.patient else "unknown",
    )

    triage_obs = (
        encounter.triage_observations[0] if encounter.triage_observations else None
    )
    if triage_obs is None:
        triage_obs = TriageObservation(
            id=uuid4(),
            encounter_id=encounter.id,
            vitals={},
            chief_complaint=None,
        )
        session.add(triage_obs)

    existing_vitals = triage_obs.vitals or {}
    vitals_dict = {**existing_vitals, **payload.model_dump(exclude={"notes"})}
    triage_obs.vitals = vitals_dict
    if payload.notes:
        triage_obs.notes = payload.notes

    patient = encounter.patient
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Encounter has no associated patient record.",
        )

    # Calculate age in years
    age_years: float | None = None
    if patient.dob is not None:
        from datetime import date

        dob = patient.dob
        if hasattr(dob, "date"):
            dob = dob.date()
        today = date.today()
        age_years = float(
            today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        )

    nurse_payload = NurseTriagePayload(
        hr=payload.hr,
        rr=payload.rr,
        sbp=payload.sbp,
        dbp=payload.dbp,
        temp_c=payload.temp_c,
        pain=payload.pain,
        age=age_years,
        chief_complaint=triage_obs.chief_complaint,
        ambulance_arrival=bool(existing_vitals.get("ambulance_arrival")),
        seen_72h=bool(existing_vitals.get("seen_72h")),
        injury=bool(existing_vitals.get("injury")),
    )

    try:
        import sys

        print(
            "[VITALS] Running nurse triage classification...",
            file=sys.stderr,
            flush=True,
        )
        logger.info("[VITALS] Running nurse triage classification...")
        triage_result = engine.classify(nurse_payload)
        print(
            f"[VITALS] Triage result: Level {triage_result.acuity_level} (model: {triage_result.model_version})",  # noqa: E501
            file=sys.stderr,
            flush=True,
        )
        print(
            f"[VITALS] Explanation: {triage_result.explanation}",
            file=sys.stderr,
            flush=True,
        )
        logger.info(
            "[VITALS] Triage result: Level %d (model: %s)",
            triage_result.acuity_level,
            triage_result.model_version,
        )
        logger.info("[VITALS] Explanation: %s", triage_result.explanation)
    except Exception as exc:  # pragma: no cover - runtime safety
        import sys

        print(
            f"[VITALS] ERROR: Nurse triage classification failed: {exc}",
            file=sys.stderr,
            flush=True,
        )
        logging.getLogger(__name__).error("Nurse triage classification failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to classify vitals with nurse triage model.",
        ) from exc

    triage_obs.triage_score = triage_result.acuity_level
    triage_obs.triage_model_version = triage_result.model_version
    encounter.acuity_level = triage_result.acuity_level

    await session.commit()
    await session.refresh(encounter)
    import sys

    print(
        f"[VITALS] Vitals recorded and triage updated successfully. Encounter: {encounter.id}, Triage Level: {encounter.acuity_level}",  # noqa: E501
        file=sys.stderr,
        flush=True,
    )
    logger.info(
        "[VITALS] Vitals recorded and triage updated successfully. Encounter: %s, Triage Level: %s",  # noqa: E501
        encounter.id,
        encounter.acuity_level,
    )

    response = NurseVitalsResponse(
        encounter_id=encounter.id,
        patient_id=encounter.patient_id,
        triage_level=triage_result.acuity_level,
        model_version=triage_result.model_version,
        explanation=triage_result.explanation,
        vitals=vitals_dict,
    )

    return StandardResponse(success=True, data=response)
