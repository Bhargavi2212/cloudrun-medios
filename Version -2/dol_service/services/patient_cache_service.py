"""
Service helpers for managing patient cache snapshots.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import FederatedPatientProfile, FederatedTimelineEvent
from dol_service.schemas.cache import PatientSnapshot, TimelineEventInput
from dol_service.schemas.portable_profile import (
    PortablePatient,
    PortableSummary,
    PortableTimelineEvent,
)


class PatientCacheService:
    """
    Store and retrieve cached patient profiles.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def ingest_snapshot(
        self,
        source_hospital_id: str,
        snapshot: PatientSnapshot,
    ) -> int:
        """
        Upsert a patient snapshot from a hospital and append timeline events.

        Returns the number of new timeline events ingested.
        """

        profile = await self.session.get(
            FederatedPatientProfile, snapshot.patient.patient_id
        )
        sanitized_patient = snapshot.patient.model_dump()
        patient_uuid = sanitized_patient.pop("patient_id")
        sanitized_patient["id"] = str(patient_uuid)
        sanitized_patient = _json_safe(sanitized_patient)
        sanitized_summaries = [
            _json_safe(summary.model_dump()) for summary in snapshot.summaries
        ]

        if profile is None:
            profile = FederatedPatientProfile(
                patient_id=patient_uuid,
                mrn=snapshot.patient.mrn,
                primary_hospital_id=source_hospital_id,
                demographics=sanitized_patient,
                summaries=sanitized_summaries or None,
                last_snapshot_at=datetime.now(tz=UTC),
            )
        else:
            profile.mrn = snapshot.patient.mrn or profile.mrn
            profile.demographics = sanitized_patient
            profile.summaries = sanitized_summaries or profile.summaries
            profile.last_snapshot_at = datetime.now(tz=UTC)
            if profile.primary_hospital_id is None:
                profile.primary_hospital_id = source_hospital_id

        self.session.add(profile)

        ingested = 0
        for event in snapshot.timeline:
            if await self._event_exists(patient_uuid, event):
                continue
            content = _json_safe(dict(event.content))
            content.pop("hospital_id", None)
            timeline_row = FederatedTimelineEvent(
                patient_id=patient_uuid,
                source_hospital_id=source_hospital_id,
                event_type=event.event_type,
                event_timestamp=event.event_timestamp,
                encounter_id=event.encounter_id,
                summary_id=event.summary_id,
                content=content,
                external_id=event.external_id,
            )
            self.session.add(timeline_row)
            ingested += 1

        await self.session.flush()
        return ingested

    async def find_patient_id_by_mrn(self, mrn: str) -> UUID | None:
        """
        Find patient_id (UUID) by MRN.
        Returns None if no patient found with the given MRN.
        """
        if not mrn:
            return None

        stmt = select(FederatedPatientProfile.patient_id).where(
            FederatedPatientProfile.mrn == mrn
        )
        result = await self.session.execute(stmt)
        patient_id = result.scalar_one_or_none()
        return patient_id

    async def get_cached_profile(
        self,
        patient_id: UUID,
    ) -> tuple[FederatedPatientProfile | None, Sequence[FederatedTimelineEvent]]:
        """
        Return cached profile and timeline events for the provided patient.
        """

        profile = await self.session.get(FederatedPatientProfile, patient_id)
        if profile is None:
            return None, []

        stmt = (
            select(FederatedTimelineEvent)
            .where(FederatedTimelineEvent.patient_id == patient_id)
            .order_by(FederatedTimelineEvent.event_timestamp.asc())
        )
        result = await self.session.execute(stmt)
        return profile, result.scalars().all()

    async def _event_exists(self, patient_id: UUID, event: TimelineEventInput) -> bool:
        if event.external_id is None:
            return False
        stmt = select(FederatedTimelineEvent.id).where(
            FederatedTimelineEvent.patient_id == patient_id,
            FederatedTimelineEvent.external_id == event.external_id,
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none() is not None

    @staticmethod
    def serialize_timeline(
        events: Sequence[FederatedTimelineEvent],
    ) -> list[PortableTimelineEvent]:
        """
        Convert timeline ORM rows into portable timeline events.
        """

        serialized: list[PortableTimelineEvent] = []
        for record in events:
            serialized.append(
                PortableTimelineEvent(
                    event_type=record.event_type,
                    encounter_id=str(record.encounter_id)
                    if record.encounter_id
                    else "",
                    summary_id=str(record.summary_id) if record.summary_id else None,
                    timestamp=record.event_timestamp,
                    content=record.content,
                )
            )
        return serialized

    @staticmethod
    def serialize_summaries(profile: FederatedPatientProfile) -> list[PortableSummary]:
        """
        Convert cached summaries into portable summaries.
        """

        raw = profile.summaries or []
        return [PortableSummary.model_validate(summary) for summary in raw]

    @staticmethod
    def serialize_patient(profile: FederatedPatientProfile) -> PortablePatient:
        """
        Convert cached demographics into a PortablePatient.
        """

        demographics = dict(profile.demographics)
        patient_identifier = demographics.get("id") or profile.patient_id
        return PortablePatient(
            id=str(patient_identifier),
            mrn=demographics.get("mrn") or (profile.mrn or ""),
            first_name=demographics.get("first_name") or "",
            last_name=demographics.get("last_name") or "",
            dob=demographics.get("dob"),
            sex=demographics.get("sex"),
            contact_info=demographics.get("contact_info"),
        )


def _json_safe(value: Any) -> Any:
    """
    Convert UUID/datetime payloads to JSON-safe structures.
    """

    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    return value
