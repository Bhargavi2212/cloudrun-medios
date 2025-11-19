"""
Pydantic schemas for Manage Agent API requests and responses.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from shared.database import ClinicalEventType


class ProfileCreateRequest(BaseModel):
    """Request schema for creating a new patient profile."""
    patient_id: str = Field(..., description="Universal patient ID in MED-{uuid4} format")
    first_name: Optional[str] = Field(None, description="Patient first name")
    last_name: Optional[str] = Field(None, description="Patient last name")
    date_of_birth: Optional[datetime] = Field(None, description="Patient date of birth")
    biological_sex: Optional[str] = Field(None, description="Biological sex (M/F/Other/Unknown)")
    active_medications: Optional[Dict[str, Any]] = Field(None, description="Current medications")
    known_allergies: Optional[Dict[str, Any]] = Field(None, description="Known allergies")
    chronic_conditions: Optional[Dict[str, Any]] = Field(None, description="Chronic conditions")
    emergency_contacts: Optional[Dict[str, Any]] = Field(None, description="Emergency contacts")


class ProfileResponse(BaseModel):
    """Response schema for patient profile data."""
    patient_id: str
    profile_version: str
    created_at: datetime
    last_updated: datetime
    first_name: Optional[str]
    last_name: Optional[str]
    date_of_birth: Optional[datetime]
    biological_sex: Optional[str]
    active_medications: Optional[Dict[str, Any]]
    known_allergies: Optional[Dict[str, Any]]
    chronic_conditions: Optional[Dict[str, Any]]
    emergency_contacts: Optional[Dict[str, Any]]
    integrity_hash: str
    
    class Config:
        from_attributes = True


class ProfileImportRequest(BaseModel):
    """Request schema for importing a portable profile."""
    encrypted_data: str = Field(..., description="Encrypted profile data")
    verification_key: Optional[str] = Field(None, description="Key for signature verification")
    import_format: str = Field("json", description="Format of imported data")


class ProfileExportResponse(BaseModel):
    """Response schema for profile export."""
    patient_id: str
    export_format: str
    encrypted_data: str
    qr_code_data: Optional[str]
    integrity_hash: str
    export_timestamp: str


class ClinicalEventCreateRequest(BaseModel):
    """Request schema for creating clinical events."""
    patient_id: str = Field(..., description="Patient ID for the event")
    timestamp: Optional[datetime] = Field(None, description="When the event occurred")
    event_type: ClinicalEventType = Field(..., description="Type of clinical event")
    clinical_summary: str = Field(..., description="Clinical summary of the event")
    structured_data: Optional[Dict[str, Any]] = Field(None, description="Structured clinical data")
    ai_generated_insights: Optional[str] = Field(None, description="AI-generated insights")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="AI confidence score")


class ClinicalEventResponse(BaseModel):
    """Response schema for clinical event data."""
    event_id: str
    patient_id: str
    timestamp: datetime
    event_type: ClinicalEventType
    clinical_summary: str
    structured_data: Optional[Dict[str, Any]]
    ai_generated_insights: Optional[str]
    confidence_score: Optional[float]
    cryptographic_signature: str
    signing_timestamp: datetime
    signing_key_fingerprint: str
    
    class Config:
        from_attributes = True


class TimelineResponse(BaseModel):
    """Response schema for patient timeline."""
    patient_id: str
    total_events: int
    events: List[ClinicalEventResponse]
    filters_applied: Dict[str, Any]


class HealthResponse(BaseModel):
    """Response schema for health checks."""
    status: str
    service: str
    version: str
    hospital_id: Optional[str] = None
    database: Optional[str] = None
    components: Optional[Dict[str, str]] = None