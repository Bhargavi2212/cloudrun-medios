"""
Summary persistence services.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.crud import AsyncCRUDRepository
from database.models import Summary


class SummaryService:
    """
    Handles CRUD operations for patient summaries.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repository = AsyncCRUDRepository[Summary](session, Summary)

    async def create_summary(
        self,
        *,
        patient_id: UUID,
        encounter_ids: list[str],
        summary_text: str,
        model_version: str,
        confidence_score: float,
    ) -> Summary:
        """
        Persist a generated summary.
        """

        payload = {
            "patient_id": patient_id,
            "encounter_ids": encounter_ids,
            "summary_text": summary_text,
            "model_version": model_version,
            "confidence_score": confidence_score,
        }
        summary = await self.repository.create(payload)
        await self.session.commit()
        await self.session.refresh(summary)
        return summary

    async def list_summaries(self, patient_id: UUID) -> Sequence[Summary]:
        """
        Retrieve summaries for a patient.
        """

        stmt = (
            select(Summary)
            .where(Summary.patient_id == patient_id)
            .order_by(Summary.created_at.desc())
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()
