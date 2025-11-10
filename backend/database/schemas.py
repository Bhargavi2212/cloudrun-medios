"""Pydantic schemas mirroring core database models."""

from __future__ import annotations

from datetime import datetime, date
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from .models import ConsultationStatus, QueueStage, SummaryType, UserStatus


class ORMBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class RoleRead(ORMBase):
    id: UUID
    name: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime


class UserRead(ORMBase):
    id: UUID
    email: EmailStr
    first_name: Optional[str]
    last_name: Optional[str]
    phone: Optional[str]
    status: UserStatus
    last_login_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime


class PatientRead(ORMBase):
    id: UUID
    mrn: str
    first_name: Optional[str]
    last_name: Optional[str]
    date_of_birth: Optional[date]
    sex: Optional[str]
    contact_phone: Optional[str]
    contact_email: Optional[str]
    created_at: datetime
    updated_at: datetime


class PatientCreate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    mrn: Optional[str] = None
    date_of_birth: Optional[date] = None
    sex: Optional[str] = None
    contact_phone: Optional[str] = None
    contact_email: Optional[str] = None


class ConsultationRead(ORMBase):
    id: UUID
    patient_id: UUID
    assigned_doctor_id: Optional[UUID]
    triage_level: Optional[int]
    status: ConsultationStatus
    scheduled_at: Optional[datetime]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    chief_complaint: Optional[str]
    reason_for_visit: Optional[str]
    created_at: datetime
    updated_at: datetime


class QueueStateRead(ORMBase):
    id: UUID
    patient_id: UUID
    patient: Optional[PatientRead] = None
    consultation_id: Optional[UUID]
    consultation: Optional[ConsultationRead] = None
    current_stage: QueueStage
    priority_level: int
    estimated_wait_seconds: Optional[int]
    assigned_to: Optional[UUID]
    created_at: datetime
    updated_at: datetime


class AudioFileRead(ORMBase):
    id: UUID
    consultation_id: Optional[UUID]
    storage_path: str
    status: str
    duration_seconds: Optional[int]
    mime_type: Optional[str]
    size_bytes: Optional[int]
    created_at: datetime


class PatientSummaryRead(ORMBase):
    id: UUID
    patient_id: UUID
    summary_type: SummaryType
    content: Optional[str]
    llm_model: Optional[str]
    created_at: datetime


class JobQueueRead(ORMBase):
    id: UUID
    task_type: str
    status: str
    priority: int
    attempts: int
    max_attempts: int
    scheduled_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]


class PaginatedResponse(ORMBase):
    items: list
    total: int = Field(ge=0)
    page: int = Field(ge=1)
    size: int = Field(ge=1)

