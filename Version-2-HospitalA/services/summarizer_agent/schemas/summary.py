"""
Summarizer request and response schemas.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class SummaryGenerateRequest(BaseModel):
    """
    Request payload to generate a longitudinal summary.
    """

    patient_id: UUID = Field(..., description="Patient identifier.")
    encounter_ids: list[UUID] = Field(
        ..., description="Encounter identifiers contributing to the summary."
    )
    highlights: list[str] | None = Field(
        default=None,
        description="Optional highlight sentences to emphasize in the summary.",
    )


class SummaryResponse(BaseModel):
    """
    Response payload representing a stored summary.
    """

    id: UUID = Field(..., description="Summary identifier.")
    patient_id: UUID = Field(..., description="Patient identifier.")
    encounter_ids: list[str] = Field(
        ..., description="Encounter IDs included in the summary."
    )
    summary_text: str = Field(..., description="Generated summary body.")
    model_version: str = Field(..., description="Model version used for generation.")
    confidence_score: float = Field(
        ..., description="Confidence score for the summary."
    )
    created_at: datetime = Field(..., description="Creation timestamp.")
    updated_at: datetime = Field(..., description="Update timestamp.")

    model_config = {
        "from_attributes": True,
    }
