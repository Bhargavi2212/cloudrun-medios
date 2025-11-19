"""
Dialogue transcript schemas.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class DialogueSegment(BaseModel):
    """
    Represents a single utterance within a transcript.
    """

    speaker: str = Field(..., description="Speaker label (e.g., doctor, patient).")
    content: str = Field(..., description="Spoken content.")


class TranscriptCreate(BaseModel):
    """
    Payload to create a dialogue transcript.
    """

    encounter_id: UUID = Field(
        ..., description="Encounter identifier the transcript belongs to."
    )
    transcript: str = Field(..., description="Full dialogue text.")
    speaker_segments: list[DialogueSegment] | None = Field(
        default=None,
        description="Optional structured segments for the dialogue.",
    )
    source: str | None = Field(
        default=None,
        description="Source of the transcript (e.g., scribe, import).",
    )


class TranscriptRead(TranscriptCreate):
    """
    Response model for dialogue transcripts.
    """

    id: UUID = Field(..., description="Transcript identifier.")
    created_at: datetime = Field(..., description="Creation timestamp.")
    updated_at: datetime = Field(..., description="Update timestamp.")

    model_config = {
        "from_attributes": True,
    }
