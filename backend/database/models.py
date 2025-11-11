"""SQLAlchemy models for Medi OS."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional, Type

from sqlalchemy import JSON, BigInteger, CheckConstraint, Column, Date, DateTime
from sqlalchemy import Enum as SAEnum
from sqlalchemy import ForeignKey, Index, Integer, Numeric, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import CITEXT, INET
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

from .base import Base, SoftDeleteMixin, TimestampMixin, UUIDPrimaryKeyMixin


def enum_values(enum_cls: Type[Enum]) -> list[str]:
    return [member.value for member in enum_cls]


# ---------- Custom Types ----------


class CIText(TypeDecorator):
    impl = String
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(CITEXT())
        return dialect.type_descriptor(String(255))

    def process_bind_param(self, value, dialect):
        return value

    def process_result_value(self, value, dialect):
        return value


class INETType(TypeDecorator):
    impl = String
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(INET())
        return dialect.type_descriptor(String(45))

    def process_bind_param(self, value, dialect):
        return value

    def process_result_value(self, value, dialect):
        return value


# ---------- Enumerations ----------


class UserStatus(str, Enum):
    ACTIVE = "active"
    PENDING = "pending"
    LOCKED = "locked"


class ConsultationStatus(str, Enum):
    INTAKE = "intake"
    TRIAGE = "triage"
    IN_PROGRESS = "in-progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class QueueStage(str, Enum):
    WAITING = "waiting"
    TRIAGE = "triage"
    SCRIBE = "scribe"
    DISCHARGE = "discharge"


class JobStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class SummaryType(str, Enum):
    LLM = "llm"
    CACHED = "cached"
    MANUAL = "manual"


class DocumentProcessingStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


class TimelineEventStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


class TimelineEventType(str, Enum):
    DOCUMENT = "document"
    LAB_RESULT = "lab_result"
    MEDICATION = "medication"
    VITALS = "vitals"
    NOTE = "note"


# ---------- IAM ----------


class Role(UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin, Base):
    __tablename__ = "roles"

    name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text())

    permissions: Mapped[List["Permission"]] = relationship(
        "Permission",
        secondary="role_permissions",
        back_populates="roles",
        lazy="joined",
    )


class Permission(UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin, Base):
    __tablename__ = "permissions"

    code: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text())

    roles: Mapped[List[Role]] = relationship(
        "Role",
        secondary="role_permissions",
        back_populates="permissions",
    )


class RolePermission(Base):
    __tablename__ = "role_permissions"

    role_id: Mapped[str] = mapped_column(ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True)
    permission_id: Mapped[str] = mapped_column(ForeignKey("permissions.id", ondelete="CASCADE"), primary_key=True)


class User(UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin, Base):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(CIText(), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(Text(), nullable=False)
    first_name: Mapped[Optional[str]] = mapped_column(String(100))
    last_name: Mapped[Optional[str]] = mapped_column(String(100))
    phone: Mapped[Optional[str]] = mapped_column(String(20))
    status: Mapped[UserStatus] = mapped_column(SAEnum(UserStatus), default=UserStatus.ACTIVE, nullable=False)
    must_reset_password: Mapped[bool] = mapped_column(default=False, nullable=False)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    roles: Mapped[List[Role]] = relationship(
        "Role",
        secondary="user_roles",
        back_populates="users",
    )


# attach back reference after class definitions
Role.users = relationship(  # type: ignore[attr-defined]
    "User",
    secondary="user_roles",
    back_populates="roles",
)


class UserRole(Base):
    __tablename__ = "user_roles"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    role_id: Mapped[str] = mapped_column(ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True)


class RefreshToken(UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin, Base):
    __tablename__ = "refresh_tokens"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token_hash: Mapped[str] = mapped_column(Text(), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked: Mapped[bool] = mapped_column(default=False, nullable=False)

    user: Mapped[User] = relationship("User", backref="refresh_tokens")

    __table_args__ = (Index("ix_refresh_tokens_user_revoked", "user_id", "revoked"),)


class AccessToken(UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin, Base):
    __tablename__ = "access_tokens"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    purpose: Mapped[str] = mapped_column(String(30), nullable=False)
    token_hash: Mapped[str] = mapped_column(Text(), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    consumed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    user: Mapped[User] = relationship("User", backref="access_tokens")

    __table_args__ = (Index("ix_access_tokens_user_purpose", "user_id", "purpose", "consumed_at"),)


class Session(UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin, Base):
    __tablename__ = "sessions"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    session_token_hash: Mapped[str] = mapped_column(Text(), nullable=False)
    ip_address: Mapped[Optional[str]] = mapped_column(INETType())
    user_agent: Mapped[Optional[str]] = mapped_column(Text())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    user: Mapped[User] = relationship("User", backref="sessions")

    __table_args__ = (Index("ix_sessions_user_expiry", "user_id", "expires_at"),)


# ---------- Patient Domain ----------


class Patient(UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin, Base):
    __tablename__ = "patients"

    mrn: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    first_name: Mapped[Optional[str]] = mapped_column(String(100))
    last_name: Mapped[Optional[str]] = mapped_column(String(100))
    date_of_birth: Mapped[Optional[datetime]] = mapped_column(Date())
    sex: Mapped[Optional[str]] = mapped_column(String(20))
    contact_phone: Mapped[Optional[str]] = mapped_column(String(30))
    contact_email: Mapped[Optional[str]] = mapped_column(String(255))
    emergency_contact: Mapped[Optional[dict]] = mapped_column(JSON)
    primary_care_provider: Mapped[Optional[str]] = mapped_column(String(255))
    additional_metadata: Mapped[Optional[dict]] = mapped_column(JSON)

    consultations: Mapped[List["Consultation"]] = relationship("Consultation", back_populates="patient")
    timeline_events: Mapped[List["TimelineEvent"]] = relationship(
        "TimelineEvent",
        back_populates="patient",
    )

    __table_args__ = (Index("ix_patients_name_dob", "last_name", "date_of_birth"),)


class Consultation(UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin, Base):
    __tablename__ = "consultations"

    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    assigned_doctor_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"))
    triage_level: Mapped[Optional[int]] = mapped_column(Integer)
    status: Mapped[ConsultationStatus] = mapped_column(SAEnum(ConsultationStatus), nullable=False)
    scheduled_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    chief_complaint: Mapped[Optional[str]] = mapped_column(Text())
    reason_for_visit: Mapped[Optional[str]] = mapped_column(Text())

    patient: Mapped[Patient] = relationship("Patient", back_populates="consultations")
    assigned_doctor: Mapped[Optional[User]] = relationship("User", backref="consultations")
    timeline_events: Mapped[List["TimelineEvent"]] = relationship(
        "TimelineEvent",
        back_populates="consultation",
    )

    __table_args__ = (
        Index("ix_consultations_patient", "patient_id"),
        Index("ix_consultations_status", "status"),
        Index("ix_consultations_doctor_status", "assigned_doctor_id", "status"),
    )


class Vital(TimestampMixin, SoftDeleteMixin, Base):
    __tablename__ = "vitals"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    consultation_id: Mapped[str] = mapped_column(ForeignKey("consultations.id", ondelete="CASCADE"), nullable=False)
    recorded_by: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"))
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    heart_rate: Mapped[Optional[int]] = mapped_column(Integer)
    respiratory_rate: Mapped[Optional[int]] = mapped_column(Integer)
    systolic_bp: Mapped[Optional[int]] = mapped_column(Integer)
    diastolic_bp: Mapped[Optional[int]] = mapped_column(Integer)
    temperature: Mapped[Optional[float]] = mapped_column(Numeric(4, 1))
    oxygen_saturation: Mapped[Optional[int]] = mapped_column(Integer)
    pain_score: Mapped[Optional[int]] = mapped_column(Integer)
    extra: Mapped[Optional[dict]] = mapped_column(JSON)

    consultation: Mapped[Consultation] = relationship("Consultation", backref="vitals")
    __table_args__ = (Index("ix_vitals_consultation_time", "consultation_id", "recorded_at"),)


class LabResult(TimestampMixin, SoftDeleteMixin, Base):
    __tablename__ = "lab_results"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    consultation_id: Mapped[Optional[str]] = mapped_column(ForeignKey("consultations.id", ondelete="CASCADE"))
    code: Mapped[Optional[str]] = mapped_column(String(50))
    description: Mapped[Optional[str]] = mapped_column(Text())
    value: Mapped[Optional[float]] = mapped_column(Numeric)
    unit: Mapped[Optional[str]] = mapped_column(String(20))
    flag: Mapped[Optional[str]] = mapped_column(String(20))
    collected_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    resulted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    source: Mapped[Optional[dict]] = mapped_column(JSON)

    consultation: Mapped[Optional[Consultation]] = relationship("Consultation", backref="lab_results")
    __table_args__ = (Index("ix_lab_results_consultation", "consultation_id", "collected_at"),)


# ---------- Queue ----------


class QueueState(UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin, Base):
    __tablename__ = "queue_states"

    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), nullable=False)
    consultation_id: Mapped[Optional[str]] = mapped_column(ForeignKey("consultations.id", ondelete="CASCADE"))
    current_stage: Mapped[QueueStage] = mapped_column(SAEnum(QueueStage), nullable=False)
    priority_level: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    estimated_wait_seconds: Mapped[Optional[int]] = mapped_column(Integer)
    assigned_to: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"))

    patient: Mapped[Patient] = relationship("Patient")
    consultation: Mapped[Optional[Consultation]] = relationship("Consultation")

    __table_args__ = (
        Index("ix_queue_states_stage_priority", "current_stage", "priority_level"),
        Index("ix_queue_states_assigned_stage", "assigned_to", "current_stage"),
    )


class QueueEvent(TimestampMixin, Base):
    __tablename__ = "queue_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    queue_state_id: Mapped[str] = mapped_column(ForeignKey("queue_states.id", ondelete="CASCADE"), nullable=False)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    previous_stage: Mapped[Optional[QueueStage]] = mapped_column(SAEnum(QueueStage))
    next_stage: Mapped[Optional[QueueStage]] = mapped_column(SAEnum(QueueStage))
    notes: Mapped[Optional[str]] = mapped_column(Text())
    created_by: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"))

    queue_state: Mapped[QueueState] = relationship("QueueState", backref="events")
    __table_args__ = (Index("ix_queue_events_state_time", "queue_state_id", "created_at"),)


# ---------- AI Scribe ----------


class AudioFile(UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin, Base):
    __tablename__ = "audio_files"

    consultation_id: Mapped[Optional[str]] = mapped_column(ForeignKey("consultations.id", ondelete="CASCADE"))
    uploaded_by: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"))
    original_filename: Mapped[Optional[str]] = mapped_column(String(255))
    storage_path: Mapped[str] = mapped_column(Text(), nullable=False)
    duration_seconds: Mapped[Optional[int]] = mapped_column(Integer)
    mime_type: Mapped[Optional[str]] = mapped_column(String(50))
    size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger)
    checksum: Mapped[Optional[str]] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(20), default="uploaded", nullable=False)

    consultation: Mapped[Optional[Consultation]] = relationship("Consultation", backref="audio_files")

    __table_args__ = (Index("ix_audio_files_consultation", "consultation_id", "created_at"),)


class Transcription(UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin, Base):
    __tablename__ = "transcriptions"

    audio_file_id: Mapped[str] = mapped_column(ForeignKey("audio_files.id", ondelete="CASCADE"), unique=True, nullable=False)
    transcribed_by: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"))
    transcription: Mapped[Optional[str]] = mapped_column(Text())
    confidence: Mapped[Optional[float]] = mapped_column(Numeric(4, 3))
    language: Mapped[Optional[str]] = mapped_column(String(10))
    is_stub: Mapped[bool] = mapped_column(default=False, nullable=False)

    audio_file: Mapped[AudioFile] = relationship("AudioFile", backref="transcription")


class Note(UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin, Base):
    __tablename__ = "notes"

    consultation_id: Mapped[str] = mapped_column(ForeignKey("consultations.id", ondelete="CASCADE"), nullable=False)
    author_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"))
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="draft")
    current_version_id: Mapped[Optional[str]] = mapped_column(ForeignKey("note_versions.id"))

    consultation: Mapped[Consultation] = relationship("Consultation", backref="notes")
    current_version: Mapped[Optional["NoteVersion"]] = relationship(
        "NoteVersion",
        foreign_keys=[current_version_id],
        post_update=True,
    )
    versions: Mapped[List["NoteVersion"]] = relationship(
        "NoteVersion",
        back_populates="note",
        foreign_keys="NoteVersion.note_id",
        cascade="all, delete-orphan",
    )

    __table_args__ = (Index("ix_notes_consultation", "consultation_id"),)


class NoteVersion(UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin, Base):
    __tablename__ = "note_versions"

    note_id: Mapped[str] = mapped_column(ForeignKey("notes.id", ondelete="CASCADE"), nullable=False)
    generated_by: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[Optional[str]] = mapped_column(Text())
    entities: Mapped[Optional[dict]] = mapped_column(JSON)
    confidence: Mapped[Optional[float]] = mapped_column(Numeric(4, 3))
    is_ai_generated: Mapped[bool] = mapped_column(default=False, nullable=False)
    created_by: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"))

    note: Mapped[Note] = relationship(
        "Note",
        back_populates="versions",
        foreign_keys=[note_id],
    )

    __table_args__ = (Index("ix_note_versions_note", "note_id", "created_at"),)


# ---------- AI Triage ----------


class TriagePrediction(TimestampMixin, SoftDeleteMixin, Base):
    __tablename__ = "triage_predictions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    consultation_id: Mapped[str] = mapped_column(ForeignKey("consultations.id", ondelete="CASCADE"), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    esi_level: Mapped[int] = mapped_column(Integer, nullable=False)
    probability: Mapped[Optional[float]] = mapped_column(Numeric(4, 3))
    feature_importances: Mapped[Optional[dict]] = mapped_column(JSON)
    inputs_snapshot: Mapped[Optional[dict]] = mapped_column(JSON)

    consultation: Mapped[Consultation] = relationship("Consultation", backref="triage_predictions")

    __table_args__ = (
        Index("ix_triage_predictions_consultation", "consultation_id"),
        Index("ix_triage_predictions_model", "model_version", "created_at"),
        CheckConstraint("esi_level BETWEEN 1 AND 5", name="ck_triage_predictions_esi_range"),
    )


# ---------- Timeline ----------


class TimelineEvent(UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin, Base):
    __tablename__ = "timeline_events"

    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    consultation_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("consultations.id", ondelete="SET NULL"),
        nullable=True,
    )
    source_file_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("files.id", ondelete="SET NULL"),
        nullable=True,
    )
    event_type: Mapped[TimelineEventType] = mapped_column(
        SAEnum(
            TimelineEventType,
            name="timelineeventtype",
            values_callable=enum_values,
        ),
        nullable=False,
    )
    event_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String(255))
    summary: Mapped[Optional[str]] = mapped_column(Text())
    data: Mapped[Optional[dict]] = mapped_column(JSON)
    status: Mapped[TimelineEventStatus] = mapped_column(
        SAEnum(
            TimelineEventStatus,
            name="timelineeventstatus",
            values_callable=enum_values,
        ),
        nullable=False,
        default=TimelineEventStatus.PENDING,
    )
    confidence: Mapped[Optional[float]] = mapped_column(Numeric(4, 3))
    notes: Mapped[Optional[str]] = mapped_column(Text())

    patient: Mapped[Patient] = relationship("Patient", back_populates="timeline_events")
    consultation: Mapped[Optional["Consultation"]] = relationship(
        "Consultation",
        back_populates="timeline_events",
    )
    source_file: Mapped[Optional["FileAsset"]] = relationship(
        "FileAsset",
        back_populates="timeline_events",
    )

    __table_args__ = (
        Index("ix_timeline_events_patient_date", "patient_id", "event_date"),
        Index("ix_timeline_events_status", "status"),
        Index("ix_timeline_events_source_file", "source_file_id"),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="ck_timeline_events_confidence_range",
        ),
    )


# ---------- AI Summaries ----------


class PatientSummary(UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin, Base):
    __tablename__ = "patient_summaries"

    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    summary_type: Mapped[SummaryType] = mapped_column(SAEnum(SummaryType), nullable=False)
    content: Mapped[Optional[str]] = mapped_column(Text())
    timeline: Mapped[Optional[dict]] = mapped_column(JSON)
    llm_model: Mapped[Optional[str]] = mapped_column(String(100))
    token_usage: Mapped[Optional[dict]] = mapped_column(JSON)
    cost_cents: Mapped[Optional[float]] = mapped_column(Numeric(8, 2))
    created_by: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"))

    patient: Mapped[Patient] = relationship("Patient", backref="summaries")

    __table_args__ = (Index("ix_patient_summaries_patient", "patient_id", "created_at"),)


class SummaryCache(Base):
    __tablename__ = "summary_cache"

    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"), primary_key=True)
    timeline_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    summary_id: Mapped[Optional[str]] = mapped_column(ForeignKey("patient_summaries.id", ondelete="SET NULL"))
    last_generated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    patient: Mapped[Patient] = relationship("Patient")
    summary: Mapped[Optional[PatientSummary]] = relationship("PatientSummary")

    __table_args__ = (Index("ix_summary_cache_expiry", "expires_at"),)


# ---------- File Storage ----------


class FileAsset(UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin, Base):
    __tablename__ = "files"

    owner_type: Mapped[str] = mapped_column(String(30), nullable=False)
    owner_id: Mapped[str] = mapped_column(String(36), nullable=False)
    original_filename: Mapped[Optional[str]] = mapped_column(String(255))
    storage_path: Mapped[str] = mapped_column(Text(), nullable=False)
    bucket: Mapped[Optional[str]] = mapped_column(String(100))
    content_type: Mapped[Optional[str]] = mapped_column(String(100))
    size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger)
    checksum: Mapped[Optional[str]] = mapped_column(String(128))
    retention_policy: Mapped[Optional[str]] = mapped_column(String(30))
    uploaded_by: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"))
    description: Mapped[Optional[str]] = mapped_column(Text())
    document_type: Mapped[Optional[str]] = mapped_column(String(50))
    status: Mapped[DocumentProcessingStatus] = mapped_column(
        SAEnum(
            DocumentProcessingStatus,
            name="documentprocessingstatus",
            values_callable=enum_values,
        ),
        nullable=False,
        default=DocumentProcessingStatus.UPLOADED,
    )
    confidence: Mapped[Optional[float]] = mapped_column(Numeric(4, 3))
    processing_metadata: Mapped[Optional[dict]] = mapped_column(JSON)
    processing_notes: Mapped[Optional[str]] = mapped_column(Text())
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[Optional[str]] = mapped_column(Text())

    timeline_events: Mapped[List["TimelineEvent"]] = relationship(
        "TimelineEvent",
        back_populates="source_file",
    )

    __table_args__ = (
        Index("ix_files_owner", "owner_type", "owner_id"),
        Index("ix_files_status", "status"),
    )


# ---------- Audit & Telemetry ----------


class AuditLog(TimestampMixin, Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"))
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    entity_type: Mapped[Optional[str]] = mapped_column(String(50))
    entity_id: Mapped[Optional[str]] = mapped_column(String(36))
    ip_address: Mapped[Optional[str]] = mapped_column(INETType())
    user_agent: Mapped[Optional[str]] = mapped_column(Text())
    payload: Mapped[Optional[dict]] = mapped_column(JSON)

    user: Mapped[Optional[User]] = relationship("User")

    __table_args__ = (
        Index("ix_audit_logs_entity", "entity_type", "entity_id"),
        Index("ix_audit_logs_user", "user_id", "created_at"),
    )


class ServiceMetric(TimestampMixin, Base):
    __tablename__ = "service_metrics"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    service_name: Mapped[str] = mapped_column(String(50), nullable=False)
    metric_name: Mapped[str] = mapped_column(String(50), nullable=False)
    metric_value: Mapped[float] = mapped_column(Numeric, nullable=False)
    metadata_payload: Mapped[Optional[dict]] = mapped_column(JSON)

    __table_args__ = (
        Index(
            "ix_service_metrics_service_metric",
            "service_name",
            "metric_name",
            "created_at",
        ),
    )


class LLMUsage(TimestampMixin, Base):
    __tablename__ = "llm_usage"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    request_id: Mapped[Optional[str]] = mapped_column(String(36))
    user_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"))
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    tokens_prompt: Mapped[Optional[int]] = mapped_column(Integer)
    tokens_completion: Mapped[Optional[int]] = mapped_column(Integer)
    cost_cents: Mapped[Optional[float]] = mapped_column(Numeric(8, 2))
    status: Mapped[str] = mapped_column(String(20), nullable=False)

    user: Mapped[Optional[User]] = relationship("User")

    __table_args__ = (Index("ix_llm_usage_model", "model", "created_at"),)


# ---------- Job Queue ----------


class JobQueue(UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin, Base):
    __tablename__ = "job_queue"

    task_type: Mapped[str] = mapped_column(String(50), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[JobStatus] = mapped_column(SAEnum(JobStatus), default=JobStatus.PENDING, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=5, nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    scheduled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[Optional[str]] = mapped_column(Text())

    __table_args__ = (
        Index("ix_job_queue_status_priority", "status", "priority", "scheduled_at"),
        Index("ix_job_queue_task_status", "task_type", "status"),
    )
