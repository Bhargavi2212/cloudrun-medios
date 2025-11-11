"""
ManageAgent DTOs (Data Transfer Objects)
Pydantic models for API requests and responses
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CheckInRequest(BaseModel):
    """Request model for patient check-in"""

    patient_id: str = Field(..., description="Patient ID (UUID)")
    chief_complaint: str = Field(..., description="Patient's chief complaint")


class VitalsSubmission(BaseModel):
    """Request model for vitals submission - permissive validation for emergency/critical cases"""

    heart_rate: int = Field(..., ge=10, le=300, description="Heart rate in BPM")
    blood_pressure_systolic: int = Field(..., ge=40, le=350, description="Systolic blood pressure")
    blood_pressure_diastolic: int = Field(..., ge=20, le=200, description="Diastolic blood pressure")
    respiratory_rate: int = Field(..., ge=4, le=60, description="Respiratory rate")
    temperature_celsius: float = Field(..., ge=25.0, le=50.0, description="Temperature in Celsius")
    oxygen_saturation: float = Field(..., ge=30.0, le=100.0, description="Oxygen saturation percentage")
    weight_kg: Optional[float] = Field(None, ge=0.5, le=500.0, description="Weight in kilograms")


class PatientQueueItem(BaseModel):
    """Response model for patient queue items"""

    queue_state_id: str
    consultation_id: str
    patient_id: str
    patient_name: str
    age: Optional[int] = None  # Patient age for display
    triage_level: Optional[int] = None
    status: str
    wait_time_minutes: int  # Time elapsed (how long they've been waiting)
    estimated_wait_minutes: Optional[int] = None  # AI prediction (how much longer)
    queue_position: Optional[int] = None
    confidence_level: Optional[str] = None  # "high", "medium", "low"
    prediction_method: Optional[str] = None  # "ml_model", "rule_based", "fallback"
    priority_score: Optional[float] = None
    assigned_doctor: Optional[str] = None
    assigned_doctor_id: Optional[str] = None  # Doctor ID for filtering
    check_in_time: datetime
    chief_complaint: str
    vitals: Optional[dict] = None  # Vitals data if available

    class Config:
        from_attributes = True


class DoctorAssignmentResponse(BaseModel):
    """Response model for doctor assignment"""

    consultation_id: str
    patient_name: str
    assigned_doctor_id: str
    assigned_doctor_name: str
    triage_level: int
    priority_score: float

    class Config:
        from_attributes = True


class WaitTimeResponse(BaseModel):
    """Response model for wait time estimation"""

    estimated_wait_minutes: int
    queue_position: int
    total_patients_in_queue: int
    confidence_level: str  # "high", "medium", "low"


class QueueResponse(BaseModel):
    """Response model for queue view"""

    patients: List[PatientQueueItem]
    total_count: int
    average_wait_time: float
    triage_distribution: dict  # Count of patients by triage level


class TriageResult(BaseModel):
    """Response model for triage calculation"""

    triage_level: int = Field(..., ge=1, le=5, description="ESI triage level (1=most urgent, 5=least urgent)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in triage assessment")
    reasoning: str = Field(..., description="Explanation for triage level")
    priority_score: float = Field(..., description="Calculated priority score")


class ConsultationRecord(BaseModel):
    """Metadata about a consultation file upload"""

    id: str
    original_filename: Optional[str]
    content_type: Optional[str]
    size_bytes: Optional[int]
    description: Optional[str]
    uploaded_at: datetime
    uploaded_by: Optional[str]
    signed_url: Optional[str]
    download_url: str
    status: str
    document_type: Optional[str]
    confidence: Optional[float]
    needs_review: bool
    processed_at: Optional[datetime]
    processing_notes: Optional[str]
    processing_metadata: Optional[Dict[str, object]]
    timeline_event_ids: Optional[List[str]] = None

    class Config:
        from_attributes = True


class DocumentStatusResponse(BaseModel):
    file_id: str
    status: str
    processed_at: Optional[datetime]
    confidence: Optional[float]
    needs_review: bool
    processing_notes: Optional[str]
    timeline_event_ids: List[str]
    metadata: Dict[str, object] = Field(default_factory=dict)


class DocumentReviewResolution(str, Enum):
    APPROVED = "approved"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


class DocumentReviewRequest(BaseModel):
    resolution: DocumentReviewResolution = Field(
        ...,
        description="Outcome of the review. Use 'approved' to clear, 'needs_review' to keep flagged, or 'failed' to reject.",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional reviewer notes stored with the file and timeline events.",
    )
    update_timeline: bool = Field(
        default=True,
        description="Whether to update associated timeline events to match the review resolution.",
    )


class TimelineSummaryResponse(BaseModel):
    summary_id: str
    summary: str
    timeline: Dict[str, Any]
    highlights: List[str]
    confidence: Optional[float]
    cached: bool
    generated_at: datetime
    model: Optional[str]
    token_usage: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        from_attributes = True


class StaffRosterItem(BaseModel):
    """Response model for staff roster items"""

    user_id: int
    full_name: str
    role: str
    is_on_duty: bool
    current_patient_load: int

    class Config:
        from_attributes = True


class StaffRosterResponse(BaseModel):
    """Response model for staff roster"""

    staff: List[StaffRosterItem]
    total_doctors: int
    total_nurses: int
    available_doctors: int
    available_nurses: int
