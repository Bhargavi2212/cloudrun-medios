"""
Transcript endpoints for the scribe-agent service.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from database.session import get_session
from services.scribe_agent.schemas import TranscriptCreate, TranscriptRead
from services.scribe_agent.services.transcript_service import TranscriptService

router = APIRouter(prefix="/scribe", tags=["transcripts"])


@router.post(
    "/transcript",
    response_model=TranscriptRead,
    status_code=status.HTTP_201_CREATED,
    summary="Store dialogue transcript",
    description="Persist a dialogue transcript captured during an encounter.",
)
async def create_transcript(
    payload: TranscriptCreate,
    session: AsyncSession = Depends(get_session),
) -> TranscriptRead:
    """
    Persist a dialogue transcript.
    """

    service = TranscriptService(session)
    transcript = await service.create_transcript(payload)
    return TranscriptRead.model_validate(transcript, from_attributes=True)


@router.get(
    "/transcript",
    response_model=list[TranscriptRead],
    summary="List dialogue transcripts",
    description="Retrieve transcripts optionally filtered by encounter.",
)
async def list_transcripts(
    encounter_id: UUID
    | None = Query(default=None, description="Filter transcripts by encounter ID."),
    session: AsyncSession = Depends(get_session),
) -> Sequence[TranscriptRead]:
    """
    Fetch dialogue transcripts.
    """

    service = TranscriptService(session)
    transcripts = await service.list_transcripts(encounter_id=encounter_id)
    return [
        TranscriptRead.model_validate(item, from_attributes=True)
        for item in transcripts
    ]
