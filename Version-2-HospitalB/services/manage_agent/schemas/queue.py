"""
Queue and check-in schemas.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class CheckInRequest(BaseModel):
    """Request to check in a patient (receptionist workflow - age + chief complaint only)."""  # noqa: E501

    patient_id: UUID | str = Field(
        ..., description="Patient identifier (UUID or string)."
    )
    chief_complaint: str = Field(
        ..., min_length=1, description="Primary reason for visit."
    )
    age: int | None = Field(
        None, description="Patient age (optional, calculated from DOB if not provided)."
    )
    injury: bool = Field(
        False, description="Whether the visit is due to an injury/accident."
    )
    ambulance_arrival: bool = Field(
        False, description="Whether the patient arrived via ambulance."
    )
    seen_72h: bool = Field(
        False,
        description="Whether the patient was seen in the hospital within the past 72 hours.",  # noqa: E501
    )

    model_config = {
        "json_encoders": {
            UUID: str,
        }
    }


class QueuePatient(BaseModel):
    """Patient in the queue."""

    queue_state_id: str = Field(..., description="Queue state identifier.")
    consultation_id: str = Field(..., description="Consultation/encounter identifier.")
    patient_id: str = Field(..., description="Patient identifier.")
    patient_name: str = Field(..., description="Full patient name.")
    age: int | None = Field(None, description="Patient age.")
    chief_complaint: str | None = Field(None, description="Chief complaint.")
    triage_level: int | None = Field(None, description="ESI triage level (1-5).")
    status: str = Field(
        ..., description="Queue status (waiting, triage, scribe, discharge)."
    )
    wait_time_minutes: int = Field(0, description="Time waiting in minutes.")
    estimated_wait_minutes: int | None = Field(None, description="Estimated wait time.")
    queue_position: int | None = Field(None, description="Position in queue.")
    confidence_level: str | None = Field(None, description="Triage confidence.")
    assigned_doctor: str | None = Field(None, description="Assigned doctor name.")
    assigned_doctor_id: str | None = Field(None, description="Assigned doctor ID.")
    priority_score: float | None = Field(None, description="Priority score.")
    prediction_method: str | None = Field(None, description="Triage prediction method.")
    check_in_time: str | None = Field(None, description="Check-in timestamp.")
    vitals: dict[str, Any] | None = Field(None, description="Vital signs.")


class ManageQueueResponse(BaseModel):
    """Queue response with patients and metrics."""

    patients: list[QueuePatient] = Field(
        default_factory=list, description="Patients in queue."
    )
    total_count: int = Field(0, description="Total patients in queue.")
    average_wait_time: float = Field(0.0, description="Average wait time in minutes.")
    triage_distribution: dict[int, int] = Field(
        default_factory=dict, description="Count by triage level."
    )

