"""
Pydantic schemas for portable profiles.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class FederatedPatientRequest(BaseModel):
    """
    Request body for federated patient queries.
    """

    patient_id: UUID = Field(..., description="Patient identifier.")


class PortableTimelineEvent(BaseModel):
    """
    Timeline event transport schema.
    """

    event_type: str = Field(
        ..., description="Type of event (encounter, triage, transcript, soap_note)."
    )
    encounter_id: str = Field(..., description="Encounter identifier.")
    summary_id: str | None = Field(
        None, description="Related summary identifier when applicable."
    )
    timestamp: datetime = Field(..., description="Event timestamp (ISO 8601).")
    content: dict[str, Any] = Field(..., description="Event content payload.")


class PortableSummary(BaseModel):
    """
    Summary payload for inclusion in portable profile.
    """

    id: str = Field(..., description="Summary identifier.")
    encounter_ids: list[str] = Field(
        ..., description="Encounters referenced by the summary."
    )
    summary_text: str = Field(..., description="Narrative summary.")
    model_version: str | None = Field(
        None, description="Model version used for generation."
    )
    confidence_score: float | None = Field(None, description="Confidence score.")
    created_at: datetime = Field(..., description="Creation timestamp.")


class PortablePatient(BaseModel):
    """
    Patient payload returned in the federated profile.
    """

    id: str = Field(..., description="Patient identifier.")
    mrn: str = Field(..., description="Medical record number.")
    first_name: str = Field(..., description="Given name.")
    last_name: str = Field(..., description="Family name.")
    dob: str | None = Field(None, description="Date of birth.")
    sex: str | None = Field(None, description="Recorded sex/gender.")
    contact_info: dict[str, Any] | None = Field(
        None, description="Contact information."
    )


class PortableProfileResponse(BaseModel):
    """
    Response payload for federated patient profile requests.
    """

    patient: PortablePatient = Field(..., description="Patient demographics.")
    timeline: list[PortableTimelineEvent] = Field(
        ..., description="Chronological event list."
    )
    summaries: list[PortableSummary] = Field(..., description="Available summaries.")
    sources: list[str] = Field(
        ..., description="Hospitals contributing to this profile."
    )


class FederatedTimelineResponse(BaseModel):
    """
    Lightweight response for peers retrieving timeline fragments.
    """

    timeline: list[PortableTimelineEvent] = Field(..., description="Timeline events.")
    summaries: list[PortableSummary] = Field(
        ..., description="Summaries associated with the patient."
    )
    source_hospital: str = Field(
        ..., description="Hospital that produced this response."
    )
