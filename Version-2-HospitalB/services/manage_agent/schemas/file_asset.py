"""
Pydantic schemas for file asset operations.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class FileAssetRead(BaseModel):
    """File asset read schema."""

    id: UUID
    patient_id: UUID | None = None
    encounter_id: UUID | None = None
    original_filename: str | None = None
    storage_path: str
    content_type: str | None = None
    size_bytes: int | None = None
    document_type: str | None = None
    upload_method: str | None = None
    status: str
    confidence: float | None = None
    extraction_status: str | None = None
    extraction_confidence: float | None = None
    confidence_tier: str | None = None
    review_status: str | None = None
    needs_manual_review: bool = False
    processing_notes: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class FileAssetCreate(BaseModel):
    """File asset create schema (for internal use)."""

    patient_id: UUID | None = None
    encounter_id: UUID | None = None
    storage_path: str
    original_filename: str | None = None
    content_type: str | None = None
    size_bytes: int | None = None
    upload_method: str | None = None
    document_type: str | None = None


class FileAssetUpdate(BaseModel):
    """File asset update schema."""

    status: str | None = None
    extraction_status: str | None = None
    extraction_confidence: float | None = None
    confidence_tier: str | None = None
    review_status: str | None = None
    needs_manual_review: bool | None = None
    processing_notes: str | None = None


class FileUploadResponse(BaseModel):
    """Response schema for file upload."""

    file_id: UUID
    original_filename: str | None = None
    size_bytes: int | None = None
    content_type: str | None = None
    status: str
    message: str = "File uploaded successfully"
