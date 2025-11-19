"""
SQLAlchemy models for Medi OS core entities.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.base import Base, TimestampMixin


class Patient(TimestampMixin, Base):
    """
    Persistent representation of a patient within a hospital.
    """

    __tablename__ = "patients"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    mrn: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    first_name: Mapped[str] = mapped_column(String(128), nullable=False)
    last_name: Mapped[str] = mapped_column(String(128), nullable=False)
    dob: Mapped[date | None] = mapped_column(Date, nullable=True)
    sex: Mapped[str | None] = mapped_column(String(16), nullable=True)
    contact_info: Mapped[dict[str, str] | None] = mapped_column(JSONB, nullable=True)

    encounters: Mapped[list[Encounter]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
    )
    summaries: Mapped[list[Summary]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
    )


class Encounter(TimestampMixin, Base):
    """
    Represents a single hospital encounter for a patient.
    """

    __tablename__ = "encounters"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    patient_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=False,
    )
    arrival_ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    disposition: Mapped[str | None] = mapped_column(String(64), nullable=True)
    location: Mapped[str | None] = mapped_column(String(128), nullable=True)
    acuity_level: Mapped[int | None] = mapped_column(Integer, nullable=True)

    patient: Mapped[Patient] = relationship(back_populates="encounters")
    triage_observations: Mapped[list[TriageObservation]] = relationship(
        back_populates="encounter",
        cascade="all, delete-orphan",
    )
    dialogue_transcripts: Mapped[list[DialogueTranscript]] = relationship(
        back_populates="encounter",
        cascade="all, delete-orphan",
    )
    soap_notes: Mapped[list[SoapNote]] = relationship(
        back_populates="encounter",
        cascade="all, delete-orphan",
    )


class TriageObservation(TimestampMixin, Base):
    """
    Captures vitals and triage decisions for an encounter.
    """

    __tablename__ = "triage_observations"
    __table_args__ = (
        UniqueConstraint("encounter_id", name="uq_triage_observations_encounter_id"),
    )

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    encounter_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("encounters.id", ondelete="CASCADE"),
        nullable=False,
    )
    vitals: Mapped[dict[str, float | int | str]] = mapped_column(JSONB, nullable=False)
    chief_complaint: Mapped[str | None] = mapped_column(String(512), nullable=True)
    notes: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    triage_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    triage_model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)

    encounter: Mapped[Encounter] = relationship(back_populates="triage_observations")


class DialogueTranscript(TimestampMixin, Base):
    """
    Stores transcribed clinician-patient dialogue associated with an encounter.
    """

    __tablename__ = "dialogue_transcripts"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    encounter_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("encounters.id", ondelete="CASCADE"),
        nullable=False,
    )
    transcript: Mapped[str] = mapped_column(Text, nullable=False)
    speaker_segments: Mapped[list[dict[str, str]]] = mapped_column(JSONB, nullable=True)
    source: Mapped[str | None] = mapped_column(String(64), nullable=True)

    encounter: Mapped[Encounter] = relationship(back_populates="dialogue_transcripts")


class SoapNote(TimestampMixin, Base):
    """
    Represents a generated or manually authored SOAP note.
    """

    __tablename__ = "soap_notes"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    encounter_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("encounters.id", ondelete="CASCADE"),
        nullable=False,
    )
    subjective: Mapped[str | None] = mapped_column(Text, nullable=True)
    objective: Mapped[str | None] = mapped_column(Text, nullable=True)
    assessment: Mapped[str | None] = mapped_column(Text, nullable=True)
    plan: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    encounter: Mapped[Encounter] = relationship(back_populates="soap_notes")


class Summary(TimestampMixin, Base):
    """
    Longitudinal view summarizing a patient's encounters.
    """

    __tablename__ = "summaries"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    patient_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=False,
    )
    encounter_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    summary_text: Mapped[str] = mapped_column(Text, nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    patient: Mapped[Patient] = relationship(back_populates="summaries")


class FileAsset(TimestampMixin, Base):
    """
    Represents an uploaded document file (PDF, image, etc.) for a patient or encounter.
    """

    __tablename__ = "file_assets"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    patient_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=True,
    )
    encounter_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("encounters.id", ondelete="CASCADE"),
        nullable=True,
    )
    original_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    document_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    upload_method: Mapped[str | None] = mapped_column(String(20), nullable=True)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="uploaded")
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    extraction_status: Mapped[str | None] = mapped_column(String(30), nullable=True)
    extraction_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_tier: Mapped[str | None] = mapped_column(String(20), nullable=True)
    review_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    needs_manual_review: Mapped[bool] = mapped_column(default=False, nullable=False)
    extraction_data: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    processing_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )
    processing_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)


class TimelineEvent(TimestampMixin, Base):
    """
    Represents a timeline event extracted from documents or generated from encounters.
    """

    __tablename__ = "timeline_events"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    patient_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=False,
    )
    encounter_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("encounters.id", ondelete="SET NULL"),
        nullable=True,
    )
    source_type: Mapped[str | None] = mapped_column(String(30), nullable=True)
    source_file_asset_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("file_assets.id", ondelete="SET NULL"),
        nullable=True,
    )
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    event_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    data: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="pending")
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    extraction_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    extraction_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )
    doctor_verified: Mapped[bool] = mapped_column(default=False, nullable=False)
    verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class AuditLog(Base):
    """
    Immutable audit trail to track federated profile access.
    """

    __tablename__ = "audit_logs"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    performed_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    details: Mapped[dict[str, str] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
