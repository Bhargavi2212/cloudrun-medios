"""
Summary generation endpoints.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from database.session import get_session
from services.summarizer_agent.core.summary import SummarizerEngine
from services.summarizer_agent.dependencies import get_summarizer_engine
from services.summarizer_agent.schemas import (
    SummaryGenerateRequest,
    SummaryResponse,
    SummaryUpdateRequest,
)
from services.summarizer_agent.services.summary_service import SummaryService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/summarizer", tags=["summaries"])


@router.post(
    "/generate-summary",
    response_model=SummaryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate patient summary",
    description="Generate and store a longitudinal summary for a patient.",
)
async def generate_summary(
    payload: SummaryGenerateRequest,
    session: AsyncSession = Depends(get_session),
    engine: SummarizerEngine = Depends(get_summarizer_engine),
) -> SummaryResponse:
    """
    Generate and persist a summary for the provided patient.
    """
    import datetime
    import sys
    import traceback

    # DIRECT FILE LOGGING for debugging - AT THE VERY START
    with open("debug_summarizer.log", "a", encoding="utf-8") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"[{datetime.datetime.now()}] ====== HANDLER FUNCTION CALLED ======\n")
        f.write(f"[{datetime.datetime.now()}] SUMMARY REQUEST RECEIVED\n")
        f.write(f"Patient: {payload.patient_id}\n")
        f.write(f"Encounters: {payload.encounter_ids}\n")
        f.write(f"Engine type: {type(engine)}\n")
        f.write(f"Engine._enabled: {hasattr(engine, '_enabled')}\n")
        f.write(f"Session type: {type(session) if session else 'None'}\n")
        f.flush()

    sys.stderr.write("[HANDLER] ====== HANDLER FUNCTION CALLED ======\n")
    sys.stderr.write(f"[HANDLER] Patient: {payload.patient_id}\n")
    sys.stderr.flush()

    logger.info("=" * 80)
    logger.info("SUMMARY GENERATION REQUEST RECEIVED")
    logger.info("Patient ID: %s", payload.patient_id)
    logger.info("Encounter IDs: %s", payload.encounter_ids)
    logger.info(
        "Engine enabled: %s, model exists: %s",
        engine._enabled,
        engine._model is not None,
    )
    logger.info("Session provided: %s", session is not None)
    logger.info("=" * 80)

    print(
        f"[SUMMARY] SUMMARY GENERATION REQUEST for patient={payload.patient_id}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[SUMMARY] Encounter IDs: {payload.encounter_ids}", file=sys.stderr, flush=True
    )
    print(
        f"[SUMMARY] Engine enabled: {engine._enabled}, model exists: {engine._model is not None}",  # noqa: E501
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[SUMMARY] Session provided: {session is not None}",
        file=sys.stderr,
        flush=True,
    )

    try:
        # Force immediate output to verify code execution
        import sys

        sys.stderr.write("[SUMMARY] ====== CALLING ENGINE.SUMMARIZE ======\n")
        sys.stderr.write(f"[SUMMARY] Session: {session is not None}\n")
        sys.stderr.write(f"[SUMMARY] Engine._enabled: {engine._enabled}\n")
        sys.stderr.write(f"[SUMMARY] Engine._model: {engine._model is not None}\n")
        sys.stderr.flush()

        logger.info("Calling engine.summarize()...")
        print("[SUMMARY] Calling engine.summarize()...", file=sys.stderr, flush=True)
        print(f"[SUMMARY] Session type: {type(session)}", file=sys.stderr, flush=True)
        print(f"[SUMMARY] Engine type: {type(engine)}", file=sys.stderr, flush=True)
        print(
            f"[SUMMARY] Engine._enabled: {engine._enabled}", file=sys.stderr, flush=True
        )
        print(
            f"[SUMMARY] Engine._model: {engine._model is not None}",
            file=sys.stderr,
            flush=True,
        )

        # Write to file immediately
        with open("debug_summarizer.log", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.datetime.now()}] About to call engine.summarize()\n")
            f.write(f"  Session: {session is not None}\n")
            f.write(f"  Engine._enabled: {engine._enabled}\n")
            f.write(f"  Engine._model: {engine._model is not None}\n")
            f.flush()

        generation = await engine.summarize(payload, session=session)

        # Force immediate output after call
        sys.stderr.write("[SUMMARY] ====== ENGINE.SUMMARIZE RETURNED ======\n")
        sys.stderr.write(f"[SUMMARY] Model version: {generation.model_version}\n")
        sys.stderr.write(f"[SUMMARY] Confidence: {generation.confidence_score}\n")
        sys.stderr.flush()

        logger.info("Generation complete. Confidence: %s", generation.confidence_score)
        logger.info("Generation model_version: %s", generation.model_version)
        logger.info(
            "Generation summary_text length: %d",
            len(generation.summary_text) if generation.summary_text else 0,
        )
        logger.info("Model version: %s", generation.model_version)
        logger.info(
            "Summary text length: %d",
            len(generation.summary_text) if generation.summary_text else 0,
        )
        logger.info("Has structured_data: %s", bool(generation.structured_data))
        logger.info(
            "Summary text preview: %s",
            (generation.summary_text[:200] + "...")
            if generation.summary_text and len(generation.summary_text) > 200
            else generation.summary_text,
        )

        print(
            f"[SUMMARY] Generation complete. Confidence: {generation.confidence_score}",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"[SUMMARY] Model version: {generation.model_version}",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"[SUMMARY] Summary text length: {len(generation.summary_text) if generation.summary_text else 0}",  # noqa: E501
            file=sys.stderr,
            flush=True,
        )
        print(
            f"[SUMMARY] Has structured_data: {bool(generation.structured_data)}",
            file=sys.stderr,
            flush=True,
        )

        with open("debug_summarizer.log", "a", encoding="utf-8") as f:
            f.write(
                f"[{datetime.datetime.now()}] Generation complete. Confidence: {generation.confidence_score}\n"  # noqa: E501
            )

        summary_service = SummaryService(session)
        summary = await summary_service.create_summary(
            patient_id=payload.patient_id,
            encounter_ids=[str(encounter_id) for encounter_id in payload.encounter_ids],
            summary_text=generation.summary_text,
            model_version=generation.model_version,
            confidence_score=generation.confidence_score,
            structured_data=generation.structured_data,
        )
        print(
            f"[SUMMARY] Summary saved with ID={summary.id}", file=sys.stderr, flush=True
        )
        with open("debug_summarizer.log", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.datetime.now()}] Summary saved with ID={summary.id}\n")
        return SummaryResponse.model_validate(summary, from_attributes=True)
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error("ERROR in generate_summary: %s", e, exc_info=True)
        logger.error("TRACEBACK:\n%s", error_trace)
        print(f"[SUMMARY] ERROR in generate_summary: {e}", file=sys.stderr, flush=True)
        print(f"[SUMMARY] TRACEBACK:\n{error_trace}", file=sys.stderr, flush=True)
        with open("debug_summarizer.log", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.datetime.now()}] ERROR: {e}\n")
            f.write(f"TRACEBACK:\n{error_trace}\n")
        raise HTTPException(
            status_code=500, detail=f"Summary generation failed: {e!s}"
        ) from e


@router.get(
    "/test", summary="Test endpoint", description="Test if summarizer agent is working"
)
async def test_endpoint():
    """Test endpoint to verify the service is working."""
    import datetime
    import sys

    sys.stderr.write("TEST ENDPOINT CALLED!\n")
    sys.stderr.flush()
    logger.info("TEST ENDPOINT CALLED!")
    print("TEST ENDPOINT CALLED!", file=sys.stderr, flush=True)
    with open("debug_summarizer.log", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now()}] TEST ENDPOINT CALLED!\n")
    return {
        "status": "ok",
        "message": "Summarizer agent is working",
        "timestamp": datetime.datetime.now().isoformat(),
    }


@router.get(
    "/history/{patient_id}",
    response_model=list[SummaryResponse],
    summary="Retrieve patient summary history",
)
async def list_summaries(
    patient_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> Sequence[SummaryResponse]:
    """
    Fetch previously generated summaries for a patient.
    """

    summary_service = SummaryService(session)
    summaries = await summary_service.list_summaries(patient_id)
    return [
        SummaryResponse.model_validate(item, from_attributes=True) for item in summaries
    ]


@router.get(
    "/summary/{summary_id}",
    response_model=SummaryResponse,
    summary="Get summary by ID",
)
async def get_summary(
    summary_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> SummaryResponse:
    """
    Retrieve a specific summary by ID.
    """
    summary_service = SummaryService(session)
    summary = await summary_service.get_summary(summary_id)
    if summary is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Summary not found"
        )
    return SummaryResponse.model_validate(summary, from_attributes=True)


@router.put(
    "/summary/{summary_id}",
    response_model=SummaryResponse,
    summary="Update summary",
)
async def update_summary(
    summary_id: UUID,
    payload: SummaryUpdateRequest,
    session: AsyncSession = Depends(get_session),
) -> SummaryResponse:
    """
    Update an existing summary.
    """
    summary_service = SummaryService(session)
    summary = await summary_service.get_summary(summary_id)
    if summary is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Summary not found"
        )

    updated = await summary_service.update_summary(
        summary,
        summary_text=payload.summary_text,
        encounter_ids=payload.encounter_ids,
    )
    return SummaryResponse.model_validate(updated, from_attributes=True)


@router.delete(
    "/summary/{summary_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete summary",
    response_model=None,
)
async def delete_summary(
    summary_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> None:
    """
    Delete a summary.
    """
    summary_service = SummaryService(session)
    summary = await summary_service.get_summary(summary_id)
    if summary is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Summary not found"
        )
    await summary_service.delete_summary(summary)
