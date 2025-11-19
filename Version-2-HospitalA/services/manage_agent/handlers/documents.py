"""
Document upload and management endpoints.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from database.session import get_session
from services.manage_agent.config import ManageAgentSettings
from services.manage_agent.dependencies import get_settings
from services.manage_agent.schemas.file_asset import FileAssetRead, FileUploadResponse
from services.manage_agent.services.file_asset_service import FileAssetService
from services.manage_agent.services.storage_service import StorageService

router = APIRouter(prefix="/manage", tags=["documents"])


def get_storage_service(
    settings: ManageAgentSettings = Depends(get_settings),
) -> StorageService:
    """Dependency to get storage service."""
    storage_root = Path(settings.storage_root)
    return StorageService(storage_root)


def get_file_asset_service(
    session: AsyncSession = Depends(get_session),
    storage_service: StorageService = Depends(get_storage_service),
) -> FileAssetService:
    """Dependency to get file asset service."""
    return FileAssetService(session, storage_service)


@router.post(
    "/documents/upload",
    response_model=FileUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload document",
    description="Upload a document file (PDF, image, etc.) for a patient or encounter.",
)
async def upload_document(
    file: UploadFile = File(...),
    patient_id: UUID | None = Form(None),
    encounter_id: UUID | None = Form(None),
    upload_method: str | None = Form(None),
    session: AsyncSession = Depends(get_session),
    file_service: FileAssetService = Depends(get_file_asset_service),
    storage_service: StorageService = Depends(get_storage_service),
) -> FileUploadResponse:
    """
    Upload a document file.

    Args:
        file: Uploaded file.
        patient_id: Optional patient ID.
        encounter_id: Optional encounter ID.
        upload_method: Upload method (camera, drag_and_drop, file_picker, etc.).
        session: Database session.
        file_service: File asset service.
        storage_service: Storage service.

    Returns:
        File upload response with file ID and metadata.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Filename is required"
        )

    # Save file to storage
    file_id, storage_path, size_bytes, _ = await storage_service.save_file(file)

    # Determine content type
    content_type = file.content_type or "application/octet-stream"

    # Determine document type from filename extension
    document_type = None
    if file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext in [".pdf"]:
            document_type = "pdf"
        elif ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]:
            document_type = "image"
        elif ext in [".txt", ".doc", ".docx"]:
            document_type = "text"

    # Create file asset record
    asset = await file_service.create_file_asset(
        patient_id=patient_id,
        encounter_id=encounter_id,
        storage_path=str(storage_path),
        original_filename=file.filename,
        content_type=content_type,
        size_bytes=size_bytes,
        upload_method=upload_method or "file_picker",
        document_type=document_type,
    )

    return FileUploadResponse(
        file_id=asset.id,
        original_filename=asset.original_filename,
        size_bytes=asset.size_bytes,
        content_type=asset.content_type,
        status=asset.status,
        message="File uploaded successfully",
    )


@router.get(
    "/documents/{file_id}",
    response_model=FileAssetRead,
    summary="Get document",
    description="Retrieve document metadata by file ID.",
)
async def get_document(
    file_id: UUID,
    file_service: FileAssetService = Depends(get_file_asset_service),
) -> FileAssetRead:
    """
    Get document metadata.

    Args:
        file_id: File asset ID.
        file_service: File asset service.

    Returns:
        File asset metadata.
    """
    asset = await file_service.get_file_asset(file_id)
    if asset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )
    return FileAssetRead.model_validate(asset, from_attributes=True)


@router.get(
    "/documents",
    response_model=list[FileAssetRead],
    summary="List documents",
    description="List documents with optional filters.",
)
async def list_documents(
    patient_id: UUID | None = None,
    encounter_id: UUID | None = None,
    status: str | None = None,
    limit: int = 100,
    offset: int = 0,
    file_service: FileAssetService = Depends(get_file_asset_service),
) -> Sequence[FileAssetRead]:
    """
    List documents with optional filters.

    Args:
        patient_id: Filter by patient ID.
        encounter_id: Filter by encounter ID.
        status: Filter by status.
        limit: Maximum number of results.
        offset: Number of results to skip.
        file_service: File asset service.

    Returns:
        List of file assets.
    """
    assets = await file_service.list_file_assets(
        patient_id=patient_id,
        encounter_id=encounter_id,
        status=status,
        limit=limit,
        offset=offset,
    )
    return [
        FileAssetRead.model_validate(asset, from_attributes=True) for asset in assets
    ]


@router.get(
    "/documents/pending-review",
    response_model=list[FileAssetRead],
    summary="List pending documents",
    description="List documents that need manual review.",
)
async def list_pending_documents(
    patient_id: UUID | None = None,
    encounter_id: UUID | None = None,
    file_service: FileAssetService = Depends(get_file_asset_service),
) -> Sequence[FileAssetRead]:
    """
    List documents that need manual review.

    Args:
        patient_id: Filter by patient ID.
        encounter_id: Filter by encounter ID.
        file_service: File asset service.

    Returns:
        List of file assets needing review.
    """
    assets = await file_service.list_file_assets(
        patient_id=patient_id,
        encounter_id=encounter_id,
        status="needs_review",
        limit=100,
        offset=0,
    )
    return [
        FileAssetRead.model_validate(asset, from_attributes=True) for asset in assets
    ]


@router.post(
    "/documents/{file_id}/confirm",
    response_model=FileAssetRead,
    summary="Confirm document",
    description="Confirm document extraction (nurse approves).",
)
async def confirm_document(
    file_id: UUID,
    notes: str | None = Form(None),
    file_service: FileAssetService = Depends(get_file_asset_service),
) -> FileAssetRead:
    """
    Confirm document extraction.

    Args:
        file_id: File asset ID.
        notes: Optional review notes.
        file_service: File asset service.

    Returns:
        Updated file asset.
    """
    asset = await file_service.get_file_asset(file_id)
    if asset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )

    updated = await file_service.update_file_asset(
        asset,
        status="completed",
        review_status="confirmed",
        needs_manual_review=False,
        processing_notes=notes or asset.processing_notes,
    )
    return FileAssetRead.model_validate(updated, from_attributes=True)


@router.post(
    "/documents/{file_id}/reject",
    response_model=FileAssetRead,
    summary="Reject document",
    description="Reject document extraction (request re-upload).",
)
async def reject_document(
    file_id: UUID,
    reason: str | None = Form(None),
    file_service: FileAssetService = Depends(get_file_asset_service),
) -> FileAssetRead:
    """
    Reject document extraction.

    Args:
        file_id: File asset ID.
        reason: Rejection reason.
        file_service: File asset service.

    Returns:
        Updated file asset.
    """
    asset = await file_service.get_file_asset(file_id)
    if asset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )

    updated = await file_service.update_file_asset(
        asset,
        status="failed",
        review_status="rejected",
        needs_manual_review=True,
        processing_notes=f"Rejected: {reason}" if reason else asset.processing_notes,
        last_error=reason or "Document extraction rejected by reviewer",
    )
    return FileAssetRead.model_validate(updated, from_attributes=True)
