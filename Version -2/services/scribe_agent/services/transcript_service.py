"""
Business logic for storing dialogue transcripts.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.crud import AsyncCRUDRepository
from database.models import DialogueTranscript
from services.scribe_agent.schemas import TranscriptCreate


class TranscriptService:
    """
    Handles persistence of dialogue transcripts.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repository = AsyncCRUDRepository[DialogueTranscript](
            session, DialogueTranscript
        )

    async def create_transcript(self, payload: TranscriptCreate) -> DialogueTranscript:
        """
        Store a new transcript.
        """

        transcript = await self.repository.create(payload.model_dump(exclude_none=True))
        await self.session.commit()
        await self.session.refresh(transcript)
        return transcript

    async def list_transcripts(
        self, encounter_id: UUID | None = None
    ) -> Sequence[DialogueTranscript]:
        """
        Retrieve transcripts optionally filtered by encounter.
        """

        if encounter_id is None:
            return await self.repository.list(limit=100)

        stmt = select(DialogueTranscript).where(
            DialogueTranscript.encounter_id == encounter_id
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()
