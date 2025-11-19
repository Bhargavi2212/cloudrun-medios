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

        # Trigger summary update after SOAP note is saved
        try:
            from sqlalchemy import select

            from database.models import Encounter

            stmt = select(Encounter).where(Encounter.id == encounter_id)
            result = await self.session.execute(stmt)
            encounter = result.scalar_one_or_none()
            if encounter:
                # Import here to avoid circular dependency
                import logging

                import httpx

                logger = logging.getLogger(__name__)

    # Try to trigger summary update (non-blocking, don't fail if summarizer
    # is unavailable)
                try:
                    # Get all encounter IDs for this patient
                    from sqlalchemy import select

                    from database.models import Encounter

                    patient_id = encounter.patient_id
                    stmt = select(Encounter.id).where(
                        Encounter.patient_id == patient_id
                    )
                    result = await self.session.execute(stmt)
                    encounter_ids = [str(eid) for eid in result.scalars().all()]

                    # Call summarizer agent to update summary
                    summarizer_url = "http://localhost:8003/summarizer/generate-summary"
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        await client.post(
                            summarizer_url,
                            json={
                                "patient_id": str(patient_id),
                                "encounter_ids": encounter_ids,
                                "highlights": [],
                            },
                        )
                    logger.info(
                        "Triggered summary update for patient %s after SOAP note creation",  # noqa: E501
                        patient_id,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to trigger summary update (non-critical): %s", e
                    )
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning("Error triggering summary update: %s", e)

        return note

    async def get_soap_note(self, note_id: UUID) -> SoapNote | None:
        """
        Retrieve a SOAP note by ID.
        """
        return await self.repository.get(note_id)

    async def update_soap_note(
        self,
        note: SoapNote,
        *,
        subjective: str | None = None,
        objective: str | None = None,
        assessment: str | None = None,
        plan: str | None = None,
    ) -> SoapNote:
        """
        Update a SOAP note.
        """
        update_data = {}
        if subjective is not None:
            update_data["subjective"] = subjective
        if objective is not None:
            update_data["objective"] = objective
        if assessment is not None:
            update_data["assessment"] = assessment
        if plan is not None:
            update_data["plan"] = plan
        updated = await self.repository.update(note, update_data)
        await self.session.commit()
        await self.session.refresh(updated)
        return updated

    async def delete_soap_note(self, note: SoapNote) -> None:
        """
        Delete a SOAP note.
        """
        await self.repository.delete(note)
        await self.session.commit()
