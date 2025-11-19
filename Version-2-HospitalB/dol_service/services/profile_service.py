"""
Services for assembling portable patient profiles.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from database.models import (
    DialogueTranscript,
    Encounter,
    Patient,
    SoapNote,
    TriageObservation,
)


class ProfileService:
    """
    Build portable profiles from local data.
    """

    def __init__(self, session: AsyncSession, hospital_id: str) -> None:
        self.session = session
        self.hospital_id = hospital_id

    async def build_profile(self, patient_id: UUID) -> dict[str, Any] | None:
        """
        Build a portable profile for the provided patient.
        """

        stmt = (
            select(Patient)
            .options(
                selectinload(Patient.encounters).selectinload(
                    Encounter.triage_observations
                ),
                selectinload(Patient.encounters).selectinload(
                    Encounter.dialogue_transcripts
                ),
                selectinload(Patient.encounters).selectinload(Encounter.soap_notes),
                selectinload(Patient.summaries),
            )
            .where(Patient.id == patient_id)
        )
        result = await self.session.execute(stmt)
        patient = result.scalars().first()
        if patient is None:
            return None

        patient_payload = {
            "id": str(patient.id),
            "mrn": patient.mrn,
            "first_name": patient.first_name,
            "last_name": patient.last_name,
            "dob": patient.dob.isoformat() if patient.dob else None,
            "sex": patient.sex,
            "contact_info": patient.contact_info,
            "hospital_id": self.hospital_id,
        }

        timeline: list[dict[str, Any]] = []
        for encounter in patient.encounters:
            timeline.append(self._build_encounter_event(encounter))
            for triage in encounter.triage_observations:
                timeline.append(self._build_triage_event(encounter, triage))
            for transcript in encounter.dialogue_transcripts:
                timeline.append(self._build_transcript_event(encounter, transcript))
            for note in encounter.soap_notes:
                timeline.append(self._build_soap_event(encounter, note))

        timeline.sort(key=lambda event: event["timestamp"])

        summaries = [
            {
                "id": str(summary.id),
                "encounter_ids": summary.encounter_ids,
                "summary_text": summary.summary_text,
                "model_version": summary.model_version,
                "confidence_score": summary.confidence_score,
                "created_at": summary.created_at.isoformat(),
                "hospital_id": self.hospital_id,
            }
            for summary in patient.summaries
        ]

        return {
            "patient": patient_payload,
            "timeline": timeline,
            "summaries": summaries,
            "source_hospital": self.hospital_id,
        }

    def _build_encounter_event(self, encounter: Encounter) -> dict[str, Any]:
        return {
            "event_type": "encounter",
            "encounter_id": str(encounter.id),
            "timestamp": _iso(encounter.arrival_ts),
            "hospital_id": self.hospital_id,
            "content": {
                "disposition": encounter.disposition,
                "acuity_level": encounter.acuity_level,
            },
        }

    def _build_triage_event(
        self, encounter: Encounter, triage: TriageObservation
    ) -> dict[str, Any]:
        return {
            "event_type": "triage",
            "encounter_id": str(encounter.id),
            "timestamp": _iso(triage.created_at),
            "hospital_id": self.hospital_id,
            "content": {
                "vitals": triage.vitals,
                "triage_score": triage.triage_score,
                "model_version": triage.triage_model_version,
            },
        }

    def _build_transcript_event(
        self, encounter: Encounter, transcript: DialogueTranscript
    ) -> dict[str, Any]:
        return {
            "event_type": "transcript",
            "encounter_id": str(encounter.id),
            "timestamp": _iso(transcript.created_at),
            "hospital_id": self.hospital_id,
            "content": {
                "transcript": transcript.transcript,
                "speaker_segments": transcript.speaker_segments,
            },
        }

    def _build_soap_event(self, encounter: Encounter, note: SoapNote) -> dict[str, Any]:
        return {
            "event_type": "soap_note",
            "encounter_id": str(encounter.id),
            "timestamp": _iso(note.created_at),
            "hospital_id": self.hospital_id,
            "content": {
                "subjective": note.subjective,
                "objective": note.objective,
                "assessment": note.assessment,
                "plan": note.plan,
                "model_version": note.model_version,
                "confidence_score": note.confidence_score,
            },
        }


def _iso(value: datetime | None) -> str:
    return value.isoformat() if value else datetime.utcnow().isoformat()
