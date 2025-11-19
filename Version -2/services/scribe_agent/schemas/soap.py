"""
SOAP note schemas.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class SoapGenerateRequest(BaseModel):
    """
    Request body for SOAP generation.
    """

    encounter_id: UUID = Field(..., description="Encounter identifier.")
    transcript: str = Field(..., description="Dialogue transcript to summarize.")


class SoapResponse(BaseModel):
    """
    Response payload for generated SOAP notes.
    """

    id: UUID = Field(..., description="SOAP note identifier.")
    encounter_id: UUID = Field(..., description="Encounter identifier.")
    subjective: str = Field(..., description="Subjective section.")
    objective: str = Field(..., description="Objective section.")
    assessment: str = Field(..., description="Assessment section.")
    plan: str = Field(..., description="Plan section.")
    model_version: str = Field(..., description="Model version used.")
    confidence_score: float | None = Field(
        default=None,
        description="Optional confidence score for generated note.",
    )
    created_at: datetime = Field(..., description="When the note was created.")
    updated_at: datetime = Field(..., description="When the note was last updated.")

    model_config = {
        "from_attributes": True,
    }


class SoapUpdateRequest(BaseModel):
    """
    Request payload to update a SOAP note.
    """

    subjective: str | None = Field(None, description="Updated subjective section.")
    objective: str | None = Field(None, description="Updated objective section.")
    assessment: str | None = Field(None, description="Updated assessment section.")
    plan: str | None = Field(None, description="Updated plan section.")
