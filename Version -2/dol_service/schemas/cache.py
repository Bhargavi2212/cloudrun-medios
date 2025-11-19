"""
Schemas for patient cache synchronization.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from dol_service.schemas.portable_profile import PortableSummary, PortableTimelineEvent


class SnapshotPatient(BaseModel):
    """
    Patient demographics pushed by a hospital.
    """

    patient_id: UUID
    mrn: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    dob: datetime | None = None
    sex: str | None = None
    contact_info: dict[str, Any] | None = None


class TimelineEventInput(BaseModel):
    """
    Timeline event payload supplied by hospitals.
    """

    external_id: str | None = None
    event_type: str
    event_timestamp: datetime
    encounter_id: UUID | None = None
    summary_id: UUID | None = None
    content: dict[str, Any] = Field(default_factory=dict)


class PatientSnapshot(BaseModel):
    """
    Snapshot describing patient demographics, summaries, and timeline events.
    """

    patient: SnapshotPatient
    summaries: list[PortableSummary] = Field(default_factory=list)
    timeline: list[TimelineEventInput] = Field(default_factory=list)


class SnapshotAck(BaseModel):
    """
    Acknowledgement returned after ingesting a snapshot.
    """

    status: str
    patient_id: UUID
    timeline_events_ingested: int


class CachedProfileResponse(BaseModel):
    """
    Aggregated profile returned to hospitals.
    """

    patient: dict[str, Any]
    timeline: list[PortableTimelineEvent]
    summaries: list[PortableSummary]
    sources: list[str]
