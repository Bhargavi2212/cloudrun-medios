"""
Queue management endpoints.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from database.models import Encounter
from database.session import get_session
from services.manage_agent.schemas.queue import ManageQueueResponse, QueuePatient

router = APIRouter(prefix="/manage", tags=["queue"])


def _calculate_age(dob: date | datetime | None) -> int | None:
    """Calculate age from date of birth."""
    if dob is None:
        return None
    today = date.today()
    if isinstance(dob, datetime):
        dob_date = dob.date()
    elif isinstance(dob, date):
        dob_date = dob
    else:
        return None
    return (
        today.year
        - dob_date.year
        - ((today.month, today.day) < (dob_date.month, dob_date.day))
    )


def _calculate_wait_time(arrival_ts: datetime) -> int:
    """Calculate wait time in minutes."""
    now = datetime.now(UTC)
    if isinstance(arrival_ts, datetime):
        delta = now - arrival_ts
        return int(delta.total_seconds() / 60)
    return 0


def _get_queue_status(encounter: Encounter) -> str:
    """Determine queue status from encounter state."""
    if encounter.disposition:
        return "discharge"
    if encounter.triage_observations:
        # Check if soap_notes relationship is loaded and has items
        try:
            if encounter.soap_notes and len(encounter.soap_notes) > 0:
                return "scribe"
        except (AttributeError, Exception):
            # Relationship not loaded or error accessing it
            pass
        return "triage"
    return "waiting"


@router.get(
    "/queue",
    response_model=ManageQueueResponse,
    summary="Get patient queue",
    description="Retrieve all patients currently in the queue with their status and triage levels.",
)
async def get_queue(
    session: AsyncSession = Depends(get_session),
) -> ManageQueueResponse:
    """
    Retrieve the current patient queue.
    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        stmt = (
            select(Encounter)
            .options(
                selectinload(Encounter.patient),
                selectinload(Encounter.triage_observations),
                selectinload(Encounter.soap_notes),
            )
            .order_by(Encounter.arrival_ts.desc())
        )
        result = await session.execute(stmt)
        encounters = result.scalars().all()

        patients: list[QueuePatient] = []
        triage_distribution: dict[int, int] = {}

        for encounter in encounters:
            try:
                if not encounter.patient:
                    continue
                patient = encounter.patient
                triage_obs = (
                    encounter.triage_observations[0]
                    if encounter.triage_observations
                    else None
                )

                triage_level = encounter.acuity_level or (
                    triage_obs.triage_score if triage_obs else None
                )
                if triage_level:
                    triage_distribution[triage_level] = (
                        triage_distribution.get(triage_level, 0) + 1
                    )

                patient_name = f"{patient.first_name} {patient.last_name}"
                age = _calculate_age(patient.dob) if patient.dob else None

                wait_time = (
                    _calculate_wait_time(encounter.arrival_ts)
                    if encounter.arrival_ts
                    else 0
                )
                queue_status = _get_queue_status(encounter)

                queue_patient = QueuePatient(
                    queue_state_id=str(encounter.id),
                    consultation_id=str(encounter.id),
                    patient_id=str(patient.id),
                    patient_name=patient_name,
                    age=age,
                    chief_complaint=triage_obs.chief_complaint if triage_obs else None,
                    triage_level=triage_level,
                    status=queue_status,
                    wait_time_minutes=wait_time,
                    estimated_wait_minutes=None,
                    check_in_time=encounter.arrival_ts.isoformat()
                    if encounter.arrival_ts
                    else None,
                    vitals=triage_obs.vitals if triage_obs else None,
                )
                patients.append(queue_patient)
            except Exception as e:
                logger.warning(f"Error processing encounter {encounter.id}: {e}")
                continue

        total_wait_time = sum(p.wait_time_minutes for p in patients)
        average_wait = total_wait_time / len(patients) if patients else 0.0

        return ManageQueueResponse(
            patients=patients,
            total_count=len(patients),
            average_wait_time=average_wait,
            triage_distribution=triage_distribution,
        )
    except Exception as e:
        logger.error(f"Error retrieving queue: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve queue: {e!s}",
        )
