"""
Service for merging local patient profiles with DOL (federated) profiles.
"""

from __future__ import annotations

import logging
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
from services.manage_agent.schemas.portable_profile import (
    PortablePatient,
    PortableProfileResponse,
    PortableSummary,
    PortableTimelineEvent,
)

logger = logging.getLogger(__name__)


class ProfileMergeService:
    """
    Merges local patient data with DOL (federated) profile data.
    """

    def __init__(self, session: AsyncSession, hospital_id: str) -> None:
        self.session = session
        self.hospital_id = hospital_id

    async def merge_profiles(
        self,
        patient_id: UUID,
        dol_profile: PortableProfileResponse | None = None,
    ) -> PortableProfileResponse:
        """
        Merge local patient profile with DOL profile (if available).

        Args:
            patient_id: Patient identifier
            dol_profile: Optional DOL profile from orchestrator

        Returns:
            Merged PortableProfileResponse with source indicators
        """
        # Build local profile
        local_profile_dict = await self._build_local_profile(patient_id)

        if local_profile_dict is None:
            # Patient not found locally - use DOL profile if available
            if dol_profile:
                logger.info(
                    "Patient %s not found locally, using DOL profile only", patient_id
                )
                return self._add_source_indicators_to_dol_profile(dol_profile)
            else:
                raise ValueError(
                    f"Patient {patient_id} not found locally and no DOL profile provided"  # noqa: E501
                )

        # Convert local profile to PortableProfileResponse format
        local_profile = self._convert_local_to_portable(local_profile_dict)

        # If no DOL profile, return local profile with source indicators
        if dol_profile is None:
            logger.info(
                "No DOL profile available for patient %s, using local profile only",
                patient_id,
            )
            return self._add_source_indicators_to_local_profile(local_profile)

        # Merge both profiles
        logger.info("Merging local and DOL profiles for patient %s", patient_id)
        return self._merge_local_and_dol(local_profile, dol_profile)

    def _convert_local_to_portable(
        self, local_profile_dict: dict[str, Any]
    ) -> PortableProfileResponse:
        """
        Convert local profile dictionary to PortableProfileResponse format.
        """
        patient_data = local_profile_dict["patient"]
        timeline_data = local_profile_dict["timeline"]
        summaries_data = local_profile_dict.get("summaries", [])

        patient = PortablePatient(
            id=patient_data["id"],
            mrn=patient_data.get("mrn", ""),
            first_name=patient_data.get("first_name", ""),
            last_name=patient_data.get("last_name", ""),
            dob=patient_data.get("dob"),
            sex=patient_data.get("sex"),
            contact_info=patient_data.get("contact_info"),
        )

        timeline = [
            PortableTimelineEvent(
                event_type=event["event_type"],
                encounter_id=event.get("encounter_id", ""),
                timestamp=event["timestamp"],
                content=event.get("content", {}),
            )
            for event in timeline_data
        ]

        summaries = [
            PortableSummary(
                id=summary["id"],
                encounter_ids=summary.get("encounter_ids", []),
                summary_text=summary.get("summary_text", ""),
                model_version=summary.get("model_version"),
                confidence_score=summary.get("confidence_score"),
                created_at=summary.get("created_at"),
            )
            for summary in summaries_data
        ]

        return PortableProfileResponse(
            patient=patient,
            timeline=timeline,
            summaries=summaries,
            sources=[self.hospital_id],
        )

    def _add_source_indicators_to_local_profile(
        self, profile: PortableProfileResponse
    ) -> PortableProfileResponse:
        """
        Add source indicators to local profile timeline events.
        """
        timeline_with_sources = []
        for event in profile.timeline:
            # Create new event with source indicator
            timeline_with_sources.append(
                PortableTimelineEvent(
                    event_type=event.event_type,
                    encounter_id=event.encounter_id,
                    timestamp=event.timestamp,
                    content=event.content,
                    source="local",
                    source_hospital_id=None,  # Local events don't have
                    # external hospital ID
                )
            )

        return PortableProfileResponse(
            patient=profile.patient,
            timeline=timeline_with_sources,
            summaries=profile.summaries,
            sources=profile.sources,
        )

    def _add_source_indicators_to_dol_profile(
        self, profile: PortableProfileResponse
    ) -> PortableProfileResponse:
        """
        Add source indicators to DOL profile timeline events.
        """
        timeline_with_sources = []
        for event in profile.timeline:
            # Create new event with source indicator
            # DOL events come from federated sources (anonymized)
            timeline_with_sources.append(
                PortableTimelineEvent(
                    event_type=event.event_type,
                    encounter_id=event.encounter_id,
                    timestamp=event.timestamp,
                    content=event.content,
                    source="federated",
                    source_hospital_id=None,  # Anonymized - hospital ID not exposed
                )
            )

        return PortableProfileResponse(
            patient=profile.patient,
            timeline=timeline_with_sources,
            summaries=profile.summaries,
            sources=profile.sources,
        )

    def _merge_local_and_dol(
        self,
        local_profile: PortableProfileResponse,
        dol_profile: PortableProfileResponse,
    ) -> PortableProfileResponse:
        """
        Merge local and DOL profiles chronologically.
        """
        # Merge patient data - prefer local (more up-to-date)
        merged_patient = local_profile.patient

        # Merge timelines chronologically
        local_timeline = [
            self._add_source_to_event(event, "local", None)
            for event in local_profile.timeline
        ]
        dol_timeline = [
            self._add_source_to_event(event, "federated", None)
            for event in dol_profile.timeline
        ]

        # Combine and sort by timestamp
        merged_timeline = local_timeline + dol_timeline
        merged_timeline.sort(key=lambda e: e.timestamp)

        # Merge summaries - combine both lists
        merged_summaries = list(local_profile.summaries) + list(dol_profile.summaries)
        # Remove duplicates based on ID
        seen_ids = set()
        unique_summaries = []
        for summary in merged_summaries:
            if summary.id not in seen_ids:
                seen_ids.add(summary.id)
                unique_summaries.append(summary)

        # Merge sources
        merged_sources = list(set(local_profile.sources + dol_profile.sources))

        logger.info(
            "Merged profiles: %d local events, %d DOL events, %d total timeline events",
            len(local_timeline),
            len(dol_timeline),
            len(merged_timeline),
        )

        return PortableProfileResponse(
            patient=merged_patient,
            timeline=merged_timeline,
            summaries=unique_summaries,
            sources=merged_sources,
        )

    def _add_source_to_event(
        self,
        event: PortableTimelineEvent,
        source: str,
        source_hospital_id: str | None,
    ) -> PortableTimelineEvent:
        """
        Add source indicator to a timeline event.
        """
        return PortableTimelineEvent(
            event_type=event.event_type,
            encounter_id=event.encounter_id,
            timestamp=event.timestamp,
            content=event.content,
            source=source,
            source_hospital_id=source_hospital_id,
        )

    async def _build_local_profile(self, patient_id: UUID) -> dict[str, Any] | None:
        """
        Build a local patient profile from the database.
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
            }
            for summary in patient.summaries
        ]

        return {
            "patient": patient_payload,
            "timeline": timeline,
            "summaries": summaries,
        }

    def _build_encounter_event(self, encounter: Encounter) -> dict[str, Any]:
        return {
            "event_type": "encounter",
            "encounter_id": str(encounter.id),
            "timestamp": self._iso(encounter.arrival_ts),
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
            "timestamp": self._iso(triage.created_at),
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
            "timestamp": self._iso(transcript.created_at),
            "content": {
                "transcript": transcript.transcript,
                "speaker_segments": transcript.speaker_segments,
            },
        }

    def _build_soap_event(self, encounter: Encounter, note: SoapNote) -> dict[str, Any]:
        return {
            "event_type": "soap_note",
            "encounter_id": str(encounter.id),
            "timestamp": self._iso(note.created_at),
            "content": {
                "subjective": note.subjective,
                "objective": note.objective,
                "assessment": note.assessment,
                "plan": note.plan,
                "model_version": note.model_version,
                "confidence_score": note.confidence_score,
            },
        }

    def _iso(self, value: datetime | None) -> str:
        return value.isoformat() if value else datetime.utcnow().isoformat()
