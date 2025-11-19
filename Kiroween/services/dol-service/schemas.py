"""
Pydantic schemas for DOL Service.

This module defines data models for portable patient profiles,
clinical timelines, and federated learning operations.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum


# Enums

class ExportFormat(str, Enum):
    """Supported profile export formats."""
    JSON = "json"
    FHIR = "fhir"
    QR_CODE = "qr_code"
    ENCRYPTED_FILE = "encrypted_file"


class PrivacyLevel(str, Enum):
    """Privacy levels for profile export."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class EventType(str, Enum):
    """Clinical event types."""
    CLINICAL_VISIT = "clinical_visit"
    EMERGENCY = "emergency"
    FOLLOW_UP = "follow_up"
    PROCEDURE = "procedure"
    LAB_RESULT = "lab_result"
    MEDICATION_CHANGE = "medication_change"
    ALLERGY_UPDATE = "allergy_update"
    DIAGNOSIS = "diagnosis"


# Core Data Models

class PatientDemographics(BaseModel):
    """Patient demographics (privacy-filtered)."""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    biological_sex: Optional[str] = None
    emergency_contact: Optional[Dict[str, str]] = None


class ClinicalEvent(BaseModel):
    """Individual clinical event."""
    event_type: EventType
    clinical_summary: str
    structured_data: Dict[str, Any] = Field(default_factory=dict)
    ai_generated_insights: Optional[str] = None
    confidence_score: Optional[float] = None


class ClinicalTimelineEntry(BaseModel):
    """Clinical timeline entry with privacy filtering."""
    entry_id: str
    patient_id: str
    timestamp: datetime
    event_type: EventType
    clinical_summary: str
    structured_data: Dict[str, Any] = Field(default_factory=dict)
    ai_generated_insights: Optional[str] = None
    confidence_score: Optional[float] = None
    cryptographic_signature: Optional[str] = None


class PortableProfile(BaseModel):
    """Complete portable patient profile."""
    patient_id: str = Field(..., description="Universal patient identifier (MED-{uuid4})")
    profile_version: str = Field(default="2.0.0", description="Profile format version")
    created_at: datetime
    last_updated: datetime
    
    # Patient information
    demographics: PatientDemographics
    
    # Clinical timeline (append-only)
    clinical_timeline: List[ClinicalTimelineEntry] = Field(default_factory=list)
    
    # Current medical status
    active_medications: List[Dict[str, Any]] = Field(default_factory=list)
    known_allergies: List[Dict[str, Any]] = Field(default_factory=list)
    chronic_conditions: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Profile integrity
    integrity_hash: Optional[str] = None
    privacy_filtered: bool = Field(default=True, description="Confirms hospital metadata removed")


# Request/Response Models

class ProfileImportRequest(BaseModel):
    """Request to import a portable profile."""
    profile_id: str
    encrypted_profile_data: str
    cryptographic_signature: str
    source_format: Optional[str] = None


class ProfileImportResponse(BaseModel):
    """Response from profile import."""
    success: bool
    patient_id: str
    local_patient_id: str
    timeline_entries_imported: int
    import_timestamp: datetime
    verification_status: str
    privacy_compliance_score: float


class ProfileExportRequest(BaseModel):
    """Request to export a portable profile."""
    patient_id: str
    export_format: ExportFormat = ExportFormat.JSON
    privacy_level: PrivacyLevel = PrivacyLevel.STANDARD
    include_full_timeline: bool = True


class ProfileExportResponse(BaseModel):
    """Response from profile export."""
    success: bool
    patient_id: str
    portable_profile_id: str
    encrypted_profile_data: str
    cryptographic_signature: str
    export_format: ExportFormat
    export_timestamp: datetime
    timeline_entries_count: int
    privacy_compliance_verified: bool


class ProfileVerificationResult(BaseModel):
    """Result of profile verification."""
    is_valid: bool
    signature_valid: bool
    tamper_detected: bool
    verification_timestamp: datetime
    signing_hospital_fingerprint: Optional[str] = None
    signature_algorithm: str = "RSA-SHA256"


# Timeline Models

class TimelineAppendRequest(BaseModel):
    """Request to append timeline entry."""
    clinical_event: ClinicalEvent
    append_timestamp: Optional[datetime] = None


class TimelineQueryRequest(BaseModel):
    """Request to search timeline."""
    query: str
    search_fields: List[str] = Field(default_factory=lambda: ["clinical_summary", "structured_data"])
    date_range: Optional[Dict[str, str]] = None
    limit: int = 50


class TimelineResponse(BaseModel):
    """Response with patient timeline."""
    patient_id: str
    timeline_entries: List[ClinicalTimelineEntry]
    total_entries: int
    date_range: Dict[str, Optional[datetime]]
    integrity_verified: bool
    privacy_filtered: bool
    query_timestamp: datetime


class TimelineSearchResponse(BaseModel):
    """Response from timeline search."""
    patient_id: str
    query: str
    results: List[ClinicalTimelineEntry]
    total_matches: int
    search_timestamp: datetime
    privacy_filtered: bool


# Federated Learning Models

class ModelParameterSubmission(BaseModel):
    """Submission of model parameters for federated learning."""
    model_type: str
    training_round: int
    model_parameters: Dict[str, Any]
    privacy_budget_used: float
    training_samples_count: int
    local_accuracy_metrics: Dict[str, float]


class ModelUpdateResponse(BaseModel):
    """Response from model parameter submission."""
    success: bool
    submission_id: str
    model_type: str
    training_round: int
    privacy_verified: bool
    signature_applied: bool
    submission_timestamp: datetime


class GlobalModelUpdate(BaseModel):
    """Global model update from federated coordinator."""
    model_type: str
    training_round: int
    aggregated_parameters: Dict[str, Any]
    participating_hospitals_count: int
    coordinator_signature: str
    aggregation_method: str = "federated_averaging"


class FederatedTrainingStatus(BaseModel):
    """Status of federated training."""
    model_type: str
    local_training_round: int
    global_training_round: int
    last_local_training: Optional[str] = None
    last_global_update: Optional[str] = None
    privacy_budget_remaining: float
    participating_hospitals_count: int
    model_accuracy_improvement: float
    federated_learning_active: bool


# Validation Models

class PrivacyValidationResult(BaseModel):
    """Result of privacy validation."""
    is_compliant: bool
    is_valid: bool = True  # For backward compatibility
    compliance_score: float
    violations: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)  # For backward compatibility


class IntegrityValidationResult(BaseModel):
    """Result of integrity validation."""
    is_valid: bool
    signature_verified: bool
    tamper_evidence: List[str] = Field(default_factory=list)
    validation_timestamp: datetime


# Peer Registry Models

class PeerHospitalInfo(BaseModel):
    """Information about a peer hospital."""
    name: str
    endpoint: str
    trust_level: str = "standard"
    capabilities: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PeerRegistrationRequest(BaseModel):
    """Request to register a peer hospital."""
    hospital_id: str
    hospital_info: PeerHospitalInfo
    certificate: Optional[str] = None
    public_key: Optional[str] = None


class PeerRegistrationResponse(BaseModel):
    """Response from peer hospital registration."""
    success: bool
    peer_hospital_id: str
    registration_timestamp: datetime
    message: str


class PeerInfo(BaseModel):
    """Peer hospital information."""
    hospital_id: str
    name: str
    endpoint: Optional[str] = None
    status: str
    trust_level: str
    capabilities: List[str] = Field(default_factory=list)
    added_at: str
    last_verified: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PeerListResponse(BaseModel):
    """Response with list of peer hospitals."""
    peers: List[PeerInfo]
    total_peers: int
    query_timestamp: datetime


# Peer Registry Models

class PeerRegistrationRequest(BaseModel):
    """Request to register a peer hospital."""
    hospital_id: str
    hospital_name: str
    public_key_fingerprint: str
    api_endpoint: str
    capabilities: List[str]
    auto_approve: bool = False


class PeerRegistrationResponse(BaseModel):
    """Response from peer hospital registration."""
    success: bool
    hospital_id: str
    status: str
    registration_timestamp: datetime
    message: str


class PeerInfo(BaseModel):
    """Information about a peer hospital."""
    hospital_id: str
    public_key_fingerprint: str
    api_endpoint: str
    capabilities: List[str]
    status: str
    trust_score: float
    registered_at: str
    last_seen: Optional[str] = None


class PeerListResponse(BaseModel):
    """Response with list of peer hospitals."""
    peers: List[PeerInfo]
    total_count: int
    filter_status: Optional[str] = None
    filter_capability: Optional[str] = None
    query_timestamp: datetime


class PeerStatusResponse(BaseModel):
    """Detailed status of a peer hospital."""
    hospital_id: str
    status: str
    capabilities: List[str]
    trust_score: float
    registered_at: datetime
    last_seen: Optional[datetime] = None
    last_communication: Optional[datetime] = None
    successful_communications: int
    failed_communications: int
    total_profiles_exchanged: int
    is_trusted: bool


class RegistryStatusResponse(BaseModel):
    """Overall registry status."""
    hospital_id: str
    total_peers: int
    active_peers: int
    pending_peers: int
    suspended_peers: int
    last_updated: datetime
    registry_healthy: bool


# Error Models

class DOLError(BaseModel):
    """DOL service error response."""
    error: str
    error_code: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime
    request_id: Optional[str] = None


class ValidationError(BaseModel):
    """Validation error details."""
    field: str
    message: str
    invalid_value: Optional[Any] = None