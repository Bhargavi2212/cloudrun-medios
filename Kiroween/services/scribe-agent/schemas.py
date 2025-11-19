"""
Pydantic schemas for Scribe Agent API requests and responses.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class SOAPNoteRequest(BaseModel):
    """Request schema for SOAP note generation."""
    patient_id: str = Field(..., description="Patient identifier")
    dialogue: Optional[str] = Field(None, description="Doctor-patient dialogue")
    patient_context: Optional[Dict[str, Any]] = Field(None, description="Patient medical context")
    structured_data: Optional[Dict[str, Any]] = Field(None, description="Structured clinical data")


class SOAPNoteResponse(BaseModel):
    """Response schema for generated SOAP notes."""
    patient_id: str
    soap_note: Dict[str, str]  # subjective, objective, assessment, plan
    confidence_score: float
    generated_at: str
    model_used: str


class ClinicalDocumentationRequest(BaseModel):
    """Request schema for clinical documentation generation."""
    patient_id: str = Field(..., description="Patient identifier")
    raw_text: str = Field(..., description="Raw clinical text to process")
    document_type: str = Field(..., description="Type of document to generate")
    patient_context: Optional[Dict[str, Any]] = Field(None, description="Patient context")
    extraction_schema: Optional[Dict[str, Any]] = Field(None, description="Schema for data extraction")


class ClinicalDocumentationResponse(BaseModel):
    """Response schema for clinical documentation."""
    patient_id: str
    document_type: str
    generated_content: str
    confidence_score: float
    generated_at: str
    model_used: str


class DialogueProcessingRequest(BaseModel):
    """Request schema for medical dialogue processing."""
    dialogue_text: str = Field(..., description="Medical dialogue to process")
    processing_type: str = Field(..., description="Type of processing (transcription, analysis, etc.)")
    patient_id: Optional[str] = Field(None, description="Associated patient ID")


class DialogueProcessingResponse(BaseModel):
    """Response schema for dialogue processing."""
    processed_dialogue: str
    extracted_entities: Dict[str, List[str]]
    confidence_score: float
    processing_type: str