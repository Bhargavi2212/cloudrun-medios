"""
Patient-related request and response models.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class PatientBase(BaseModel):
    """
    Common attributes shared across patient operations.
    """

    mrn: str | None = Field(
        None, description="Medical record number assigned by the hospital."
    )
    first_name: str = Field(..., description="Patient given name.")
    last_name: str = Field(..., description="Patient family name.")
    dob: date | None = Field(None, description="Date of birth.")
    sex: str | None = Field(None, description="Administrative sex or gender.")
    contact_info: dict[str, Any] | None = Field(
        None,
        description="Contact details for the patient (phone, email, address).",
    )


class PatientCreate(PatientBase):
    """
    Payload for creating a patient.
    """


class PatientUpdate(BaseModel):
    """
    Payload for mutating a patient.
    """

    first_name: str | None = Field(None, description="Updated given name.")
    last_name: str | None = Field(None, description="Updated family name.")
    dob: date | None = Field(None, description="Updated date of birth.")
    sex: str | None = Field(None, description="Updated sex or gender value.")
    contact_info: dict[str, Any] | None = Field(
        None,
        description="Updated contact details.",
    )


class PatientRead(PatientBase):
    """
    API response model representing a patient.
    """

    id: UUID = Field(..., description="Unique patient identifier (UUID).")
    created_at: datetime = Field(..., description="Timestamp of creation.")
    updated_at: datetime = Field(..., description="Timestamp of last update.")

    model_config = {
        "from_attributes": True,
    }
