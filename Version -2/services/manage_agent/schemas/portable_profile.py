"""
Pydantic schemas mirroring the DOL portable profile response for check-in.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PortableTimelineEvent(BaseModel):
    """
    Timeline event returned from the DOL or local database.
    """

    event_type: str = Field(
        ..., description="Type of event (encounter, triage, transcript, soap_note)."
    )
    encounter_id: str = Field(..., description="Encounter identifier.")
    timestamp: datetime = Field(..., description="Event timestamp.")
    content: dict[str, Any] = Field(..., description="Event payload.")
    source: str | None = Field(
        None, description="Source of event: 'local' or 'federated'."
    )
    source_hospital_id: str | None = Field(
        None, description="Anonymized hospital ID for federated events, null for local."
    )


class PortableSummary(BaseModel):
    """
    Summary data accompanying the portable profile.
    """

    id: str = Field(..., description="Summary identifier.")
    encounter_ids: list[str] = Field(
        ..., description="Encounter IDs referenced by the summary."
    )
    summary_text: str = Field(..., description="Summary narrative text.")
    model_version: str | None = Field(
        None, description="Model version used to generate the summary."
    )
    confidence_score: float | None = Field(
        None, description="Model confidence score, if available."
    )
    created_at: datetime = Field(..., description="Creation timestamp.")


class PortablePatient(BaseModel):
    """
    Patient demographics as returned in the portable profile.
    """

    id: str = Field(..., description="Patient identifier.")
    mrn: str = Field(..., description="Patient medical record number.")
    first_name: str = Field(..., description="Given name.")
    last_name: str = Field(..., description="Family name.")
    dob: str | None = Field(None, description="Date of birth.")
    sex: str | None = Field(None, description="Sex/gender value.")
    contact_info: dict[str, Any] | None = Field(
        None, description="Optional contact metadata."
    )


class PortableProfileResponse(BaseModel):
    """
    Aggregated portable profile returned to the frontend.
    """

    patient: PortablePatient = Field(..., description="Patient information.")
    timeline: list[PortableTimelineEvent] = Field(
        ..., description="Chronological timeline events."
    )
    summaries: list[PortableSummary] = Field(
        ..., description="Summaries associated with the patient."
    )
    sources: list[str] = Field(
        ..., description="Hospital sources contributing to the profile."
    )
