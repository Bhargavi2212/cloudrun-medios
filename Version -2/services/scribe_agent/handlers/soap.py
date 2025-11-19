"""
SOAP generation endpoints.
"""

from __future__ import annotations

from datetime import date
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from database.models import Encounter, SoapNote
from database.session import get_session
from services.scribe_agent.core.soap import ScribeEngine
from services.scribe_agent.dependencies import get_scribe_engine
from services.scribe_agent.schemas import (
    SoapGenerateRequest,
    SoapResponse,
    SoapUpdateRequest,
)
from services.scribe_agent.services.soap_service import SoapService

router = APIRouter(prefix="/scribe", tags=["soap"])


@router.get("/test")
async def test_endpoint():
    """Simple test endpoint to verify requests are reaching the service."""
    import sys

    print("✅ TEST ENDPOINT CALLED!", file=sys.stderr, flush=True)
    import logging

    logger = logging.getLogger(__name__)
    logger.info("✅ TEST ENDPOINT CALLED!")
    return {"status": "ok", "message": "Scribe-agent is responding!"}


def _calculate_age(dob: date | None) -> int | None:
    """Calculate age from date of birth."""
    if dob is None:
        return None
    today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


@router.post(
    "/generate-soap",
    response_model=SoapResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate SOAP note",
    description="Generate a SOAP note from a dialogue transcript and persist it.",
)
async def generate_soap_note(
    payload: SoapGenerateRequest,
    session: AsyncSession = Depends(get_session),
    engine: ScribeEngine = Depends(get_scribe_engine),
) -> SoapResponse:
    """
    Generate and store a SOAP note for the provided encounter.
    """
    import datetime
    import logging
    import sys
    import traceback

    # DIRECT FILE LOGGING for debugging
    with open("debug_scribe.log", "a", encoding="utf-8") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"[{datetime.datetime.now()}] SOAP REQUEST RECEIVED\n")
        f.write(f"Encounter: {payload.encounter_id}\n")
        f.write(
            f"Transcript len: {len(payload.transcript) if payload.transcript else 0}\n"
        )
        f.write(
            f"Transcript preview: {payload.transcript[:100] if payload.transcript else 'None'}\n"  # noqa: E501
        )

    logger = logging.getLogger(__name__)

    try:
        print("=" * 80, file=sys.stderr, flush=True)
        print(
            "🔵🔵🔵 SOAP GENERATION REQUEST RECEIVED 🔵🔵🔵",
            file=sys.stderr,
            flush=True,
        )
        print(f"Encounter ID: {payload.encounter_id}", file=sys.stderr, flush=True)
        print(
            f"Transcript length: {len(payload.transcript) if payload.transcript else 0} chars",  # noqa: E501
            file=sys.stderr,
            flush=True,
        )
        print(
            f"Transcript preview: {payload.transcript[:200] if payload.transcript else 'EMPTY'}",  # noqa: E501
            file=sys.stderr,
            flush=True,
        )
        print("=" * 80, file=sys.stderr, flush=True)

        logger.info(
            "🔵 SOAP generation request received for encounter=%s", payload.encounter_id
        )
        logger.info(
            "🔵 Transcript length: %d chars",
            len(payload.transcript) if payload.transcript else 0,
        )
        logger.info(
            "🔵 Transcript preview: %s",
            payload.transcript[:200] if payload.transcript else "EMPTY",
        )
    except Exception as e:
        print(f"❌ ERROR in logging: {e}", file=sys.stderr, flush=True)
        print(traceback.format_exc(), file=sys.stderr, flush=True)

    # Gather patient context for better SOAP generation
    context = {}
    try:
        stmt = (
            select(Encounter)
            .where(Encounter.id == payload.encounter_id)
            .options(
                selectinload(Encounter.patient),
                selectinload(Encounter.triage_observations),
            )
        )
        result = await session.execute(stmt)
        encounter = result.scalar_one_or_none()

        if encounter and encounter.patient:
            patient = encounter.patient
            context["age"] = _calculate_age(patient.dob)

            # Get triage observations for vitals and chief complaint
            if encounter.triage_observations:
                triage = encounter.triage_observations[0]
                if triage.chief_complaint:
                    context["chief_complaint"] = triage.chief_complaint
                if triage.vitals:
                    context["vitals"] = triage.vitals
    except Exception as e:
        # If context gathering fails, continue without it
        import logging

        logger = logging.getLogger(__name__)
        logger.warning("Failed to gather patient context for SOAP generation: %s", e)

    print("🔵 Calling engine.generate()...", file=sys.stderr, flush=True)
    logger.info("🔵 Calling engine.generate()...")
    generation = await engine.generate(payload, context=context if context else None)
    print(
        f"🔵 Generation complete. Model version: {generation.model_version}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"🔵 Subjective length: {len(generation.subjective)} chars",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"🔵 Objective length: {len(generation.objective)} chars",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"🔵 Model version contains 'gemini': {'gemini' in generation.model_version}",
        file=sys.stderr,
        flush=True,
    )
    logger.info("🔵 Generation complete. Model version: %s", generation.model_version)
    logger.info("🔵 Subjective length: %d chars", len(generation.subjective))
    logger.info("🔵 Objective length: %d chars", len(generation.objective))

    service = SoapService(session)
    note = await service.save_generated_note(
        encounter_id=payload.encounter_id,
        subjective=generation.subjective,
        objective=generation.objective,
        assessment=generation.assessment,
        plan=generation.plan,
        model_version=generation.model_version,
        confidence_score=0.85 if "gemini" in generation.model_version else 0.5,
    )
    print(
        f"🔵 SOAP note saved with ID={note.id}, model_version={note.model_version}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"🔵 Subjective preview: {note.subjective[:100]}...",
        file=sys.stderr,
        flush=True,
    )
    logger.info("🔵 SOAP note saved with ID=%s", note.id)
    return SoapResponse.model_validate(note, from_attributes=True)


@router.get(
    "/soap/{note_id}",
    response_model=SoapResponse,
    summary="Get SOAP note by ID",
)
async def get_soap_note(
    note_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> SoapResponse:
    """
    Retrieve a specific SOAP note by ID.
    """
    service = SoapService(session)
    note = await service.get_soap_note(note_id)
    if note is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="SOAP note not found"
        )
    return SoapResponse.model_validate(note, from_attributes=True)


@router.get(
    "/soap/encounter/{encounter_id}",
    response_model=list[SoapResponse],
    summary="List SOAP notes for encounter",
)
async def list_soap_notes(
    encounter_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> list[SoapResponse]:
    """
    Retrieve all SOAP notes for an encounter.
    """
    stmt = (
        select(SoapNote)
        .where(SoapNote.encounter_id == encounter_id)
        .order_by(SoapNote.created_at.desc())
    )
    result = await session.execute(stmt)
    notes = result.scalars().all()
    return [SoapResponse.model_validate(note, from_attributes=True) for note in notes]


@router.put(
    "/soap/{note_id}",
    response_model=SoapResponse,
    summary="Update SOAP note",
)
async def update_soap_note(
    note_id: UUID,
    payload: SoapUpdateRequest,
    session: AsyncSession = Depends(get_session),
) -> SoapResponse:
    """
    Update an existing SOAP note.
    """
    service = SoapService(session)
    note = await service.get_soap_note(note_id)
    if note is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="SOAP note not found"
        )

    updated = await service.update_soap_note(
        note,
        subjective=payload.subjective,
        objective=payload.objective,
        assessment=payload.assessment,
        plan=payload.plan,
    )
    return SoapResponse.model_validate(updated, from_attributes=True)


@router.delete(
    "/soap/{note_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete SOAP note",
    response_model=None,
)
async def delete_soap_note(
    note_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> None:
    """
    Delete a SOAP note.
    """
    service = SoapService(session)
    note = await service.get_soap_note(note_id)
    if note is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="SOAP note not found"
        )
    await service.delete_soap_note(note)
