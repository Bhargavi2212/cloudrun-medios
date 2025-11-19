"""
SOAP note persistence logic.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from database.crud import AsyncCRUDRepository
from database.models import SoapNote


class SoapService:
    """
    Handles persistence of SOAP notes.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repository = AsyncCRUDRepository[SoapNote](session, SoapNote)

    async def save_generated_note(
        self,
        *,
        encounter_id: UUID,
        subjective: str,
        objective: str,
        assessment: str,
        plan: str,
        model_version: str,
        confidence_score: float | None = None,
    ) -> SoapNote:
        """
        Persist a generated SOAP note.
        """

        payload = {
            "encounter_id": encounter_id,
            "subjective": subjective,
            "objective": objective,
            "assessment": assessment,
            "plan": plan,
            "model_version": model_version,
            "confidence_score": confidence_score,
        }
        note = await self.repository.create(payload)
        await self.session.commit()
        await self.session.refresh(note)
        return note
