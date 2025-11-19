"""
Schemas for hospital registry management.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, HttpUrl


class HospitalRegistration(BaseModel):
    """
    Registration or heartbeat payload from a hospital.
    """

    hospital_id: str = Field(..., min_length=3, max_length=64)
    name: str = Field(..., min_length=1, max_length=255)
    manage_url: HttpUrl
    scribe_url: HttpUrl | None = None
    summarizer_url: HttpUrl | None = None
    dol_url: HttpUrl | None = None
    capabilities: list[str] = Field(default_factory=list)


class HospitalRegistryRead(BaseModel):
    """
    Public view of a registered hospital.
    """

    id: str
    name: str
    manage_url: str
    scribe_url: str | None = None
    summarizer_url: str | None = None
    dol_url: str | None = None
    status: str
    capabilities: list[str] | None = None
    last_seen_at: datetime

    class Config:
        from_attributes = True
