"""
Database models for portable patient profiles and federated learning.

This module defines the core data models for Medi OS v2.0:
- PortableProfile: Patient-controlled medical passports
- ClinicalEvent: Individual clinical events in patient timeline
- LocalPatientRecord: Hospital-specific records (full fidelity)
- Privacy-preserving data structures with zero hospital metadata
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    JSON,
    CheckConstraint,
    Index,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .base import Base


class ClinicalEventType(str, Enum):
    """Types of clinical events in patient timeline."""
    VISIT = "visit"
    PROCEDURE = "procedure"
    DIAGNOSIS = "diagnosis"
    MEDICATION = "medication"
    ALLERGY = "allergy"
    LAB_RESULT = "lab_result"
    IMAGING = "imaging"
    SUMMARY = "summary"
    EMERGENCY = "emergency"


class ProfileSyncStatus(str, Enum):
    """Status of profile synchronization."""
    SYNCED = "synced"
    PENDING = "pending"
    CONFLICT = "conflict"
    ERROR = "error"


class PortableProfile(Base):
    """
    Patient-controlled portable medical profile.
    
    This model represents the core portable profile that patients carry
    with them. It contains ONLY clinical data with zero hospital metadata
    to ensure absolute privacy.
    """
    __tablename__ = "portable_profiles"
    
    # Primary identifier - MED-{uuid4} format
    patient_id: Mapped[str] = mapped_column(
        String(40),
        primary_key=True,
        comment="Universal patient ID in MED-{uuid4} format"
    )
    
    # Profile metadata
    profile_version: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="2.0.0",
        comment="Profile format version for compatibility"
    )
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Profile creation timestamp"
    )
    
    last_updated: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="Last profile update timestamp"
    )
    
    # Patient demographics (minimal for privacy)
    first_name: Mapped[Optional[str]] = mapped_column(
        String(100),
        comment="Patient first name (optional for privacy)"
    )
    
    last_name: Mapped[Optional[str]] = mapped_column(
        String(100),
        comment="Patient last name (optional for privacy)"
    )
    
    date_of_birth: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        comment="Patient date of birth"
    )
    
    biological_sex: Mapped[Optional[str]] = mapped_column(
        String(20),
        comment="Biological sex for medical purposes"
    )
    
    # Current medical status (structured data)
    active_medications: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        comment="Current medications with dosages"
    )
    
    known_allergies: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        comment="Known allergies and reactions"
    )
    
    chronic_conditions: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        comment="Chronic medical conditions"
    )
    
    emergency_contacts: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        comment="Emergency contact information"
    )
    
    # Profile integrity and security
    integrity_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="SHA-256 hash for profile integrity verification"
    )
    
    encryption_key_fingerprint: Mapped[Optional[str]] = mapped_column(
        String(64),
        comment="Fingerprint of encryption key for profile"
    )
    
    # Relationships
    clinical_events: Mapped[List["ClinicalEvent"]] = relationship(
        "ClinicalEvent",
        back_populates="profile",
        cascade="all, delete-orphan",
        order_by="ClinicalEvent.timestamp.desc()"
    )
    
    profile_signatures: Mapped[List["ProfileSignature"]] = relationship(
        "ProfileSignature",
        back_populates="profile",
        cascade="all, delete-orphan"
    )
    
    # Constraints
    __table_args__ = (
        CheckConstraint(
            "patient_id ~ '^MED-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'",
            name="ck_patient_id_format"
        ),
        CheckConstraint(
            "biological_sex IN ('M', 'F', 'Other', 'Unknown')",
            name="ck_biological_sex_values"
        ),
        Index("ix_portable_profiles_last_updated", "last_updated"),
        Index("ix_portable_profiles_dob", "date_of_birth"),
    )


class ClinicalEvent(Base):
    """
    Individual clinical event in patient timeline.
    
    Represents a single clinical event (visit, procedure, diagnosis, etc.)
    with privacy-filtered content and cryptographic integrity.
    """
    __tablename__ = "clinical_events"
    
    # Primary key
    event_id: Mapped[str] = mapped_column(
        String(40),
        primary_key=True,
        default=lambda: f"EVT-{uuid.uuid4()}",
        comment="Unique event identifier"
    )
    
    # Foreign key to portable profile
    patient_id: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
        comment="Reference to portable profile"
    )
    
    # Event metadata
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="When the clinical event occurred"
    )
    
    event_type: Mapped[ClinicalEventType] = mapped_column(
        nullable=False,
        comment="Type of clinical event"
    )
    
    # Clinical content (privacy-filtered)
    clinical_summary: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Clinical summary of the event"
    )
    
    structured_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        comment="Structured clinical data (vitals, lab values, etc.)"
    )
    
    ai_generated_insights: Mapped[Optional[str]] = mapped_column(
        Text,
        comment="AI-generated clinical insights"
    )
    
    confidence_score: Mapped[Optional[float]] = mapped_column(
        Float,
        comment="Confidence score for AI-generated content"
    )
    
    # Tamper evidence (no institutional identification)
    cryptographic_signature: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="Digital signature for tamper detection"
    )
    
    signing_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="When the event was signed"
    )
    
    signing_key_fingerprint: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="Fingerprint of signing key (verifiable but not identifiable)"
    )
    
    # Relationships
    profile: Mapped["PortableProfile"] = relationship(
        "PortableProfile",
        back_populates="clinical_events"
    )
    
    # Constraints
    __table_args__ = (
        CheckConstraint(
            "confidence_score >= 0.0 AND confidence_score <= 1.0",
            name="ck_confidence_score_range"
        ),
        Index("ix_clinical_events_patient_timestamp", "patient_id", "timestamp"),
        Index("ix_clinical_events_type", "event_type"),
        Index("ix_clinical_events_timestamp", "timestamp"),
    )


class LocalPatientRecord(Base):
    """
    Hospital-specific patient record with full fidelity.
    
    This model stores complete hospital records including all metadata
    that is NOT included in portable profiles. This data stays local
    to each hospital and is never exported.
    """
    __tablename__ = "local_patient_records"
    
    # Primary key
    local_record_id: Mapped[str] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Hospital-specific record ID"
    )
    
    # Link to portable profile
    portable_patient_id: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
        comment="Links to MED-{uuid4} in portable profile"
    )
    
    # Hospital-specific identifiers (NEVER exported)
    hospital_id: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Hospital identifier (never in portable profile)"
    )
    
    hospital_mrn: Mapped[Optional[str]] = mapped_column(
        String(50),
        comment="Hospital medical record number"
    )
    
    # Administrative data (NEVER exported)
    admission_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        comment="Hospital admission date"
    )
    
    discharge_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        comment="Hospital discharge date"
    )
    
    attending_physician: Mapped[Optional[str]] = mapped_column(
        String(100),
        comment="Attending physician name (never exported)"
    )
    
    department: Mapped[Optional[str]] = mapped_column(
        String(100),
        comment="Hospital department"
    )
    
    room_number: Mapped[Optional[str]] = mapped_column(
        String(20),
        comment="Hospital room number"
    )
    
    # Billing and insurance (NEVER exported)
    insurance_info: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        comment="Insurance and billing information"
    )
    
    billing_codes: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        comment="ICD-10, CPT, and other billing codes"
    )
    
    # Complete clinical data (full fidelity)
    detailed_clinical_notes: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        comment="Complete clinical notes with full detail"
    )
    
    complete_lab_results: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        comment="Complete lab results with reference ranges"
    )
    
    imaging_studies: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        comment="Imaging studies and reports"
    )
    
    # Profile synchronization
    last_profile_import: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        comment="When profile was last imported"
    )
    
    last_profile_export: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        comment="When profile was last exported"
    )
    
    profile_sync_status: Mapped[ProfileSyncStatus] = mapped_column(
        nullable=False,
        default=ProfileSyncStatus.SYNCED,
        comment="Profile synchronization status"
    )
    
    # Record metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    # Constraints
    __table_args__ = (
        UniqueConstraint(
            "hospital_id", "portable_patient_id",
            name="uq_hospital_patient"
        ),
        UniqueConstraint(
            "hospital_id", "hospital_mrn",
            name="uq_hospital_mrn"
        ),
        Index("ix_local_records_hospital_patient", "hospital_id", "portable_patient_id"),
        Index("ix_local_records_admission", "admission_date"),
    )


class ProfileSignature(Base):
    """
    Cryptographic signature for profile integrity.
    
    Provides tamper evidence for portable profiles while maintaining
    hospital anonymity through verifiable but non-identifiable signatures.
    """
    __tablename__ = "profile_signatures"
    
    # Primary key
    signature_id: Mapped[str] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique signature identifier"
    )
    
    # Foreign key to portable profile
    patient_id: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
        comment="Reference to portable profile"
    )
    
    # Signature metadata
    signature_algorithm: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="RSA-SHA256",
        comment="Cryptographic signature algorithm"
    )
    
    signature_value: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Base64-encoded signature value"
    )
    
    signing_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="When signature was created"
    )
    
    # Verifiable but not identifiable
    public_key_fingerprint: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="SHA-256 fingerprint of public key for verification"
    )
    
    # What was signed
    signed_content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="SHA-256 hash of signed content"
    )
    
    signature_scope: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="What part of profile was signed (full_profile, clinical_event, etc.)"
    )
    
    # Relationships
    profile: Mapped["PortableProfile"] = relationship(
        "PortableProfile",
        back_populates="profile_signatures"
    )
    
    # Constraints
    __table_args__ = (
        CheckConstraint(
            "signature_algorithm IN ('RSA-SHA256', 'ECDSA-SHA256', 'Ed25519')",
            name="ck_signature_algorithm"
        ),
        Index("ix_signatures_patient_timestamp", "patient_id", "signing_timestamp"),
        Index("ix_signatures_fingerprint", "public_key_fingerprint"),
    )