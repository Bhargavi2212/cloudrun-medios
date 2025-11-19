"""
Summary generation endpoints.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from database.session import get_session
from services.summarizer_agent.core.summary import SummarizerEngine
from services.summarizer_agent.dependencies import get_summarizer_engine
from services.summarizer_agent.schemas import SummaryGenerateRequest, SummaryResponse
from services.summarizer_agent.services.summary_service import SummaryService

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

    generation = await engine.summarize(payload)
    summary_service = SummaryService(session)
    summary = await summary_service.create_summary(
        patient_id=payload.patient_id,
        encounter_ids=[str(encounter_id) for encounter_id in payload.encounter_ids],
        summary_text=generation.summary_text,
        model_version=generation.model_version,
        confidence_score=generation.confidence_score,
    )
    return SummaryResponse.model_validate(summary, from_attributes=True)


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
