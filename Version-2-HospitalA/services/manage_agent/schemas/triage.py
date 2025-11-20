"""
Triage classification schemas.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TriageRequest(BaseModel):
    """
    Request payload capturing vital signs for triage classification.
    """

    hr: int = Field(..., description="Heart rate (beats per minute).")
    rr: int = Field(..., description="Respiratory rate (breaths per minute).")
    sbp: int = Field(..., description="Systolic blood pressure (mmHg).")
    dbp: int = Field(..., description="Diastolic blood pressure (mmHg).")
    temp_c: float = Field(..., description="Body temperature (°C).")
    spo2: int = Field(..., description="Pulse oximetry reading (percentage).")
    pain: int = Field(..., ge=0, le=10, description="Pain score on 0-10 scale.")


class TriageResponse(BaseModel):
    """
    Response payload describing the triage acuity.
    """

    acuity_level: int = Field(
        ..., description="Computed acuity level (1=Critical, 5=Routine)."
    )
    model_version: str = Field(
        ..., description="Model version producing the prediction."
    )
    explanation: str = Field(
        ..., description="Human-readable explanation of classification."
    )


class NurseVitalsRequest(TriageRequest):
    """
    Request payload for nurse triage with vital signs.
    Extends TriageRequest for nurse-specific triage.
    """
    pass


class NurseVitalsResponse(TriageResponse):
    """
    Response payload for nurse triage classification.
    Extends TriageResponse for nurse-specific triage.
    """
    pass
