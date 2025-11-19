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
    patient_id: str | None = Form(None),
    encounter_id: str | None = Form(None),
    upload_method: str | None = Form(None),
    session: AsyncSession = Depends(get_session),
    file_service: FileAssetService = Depends(get_file_asset_service),
    storage_service: StorageService = Depends(get_storage_service),
) -> FileUploadResponse:
    """
    Upload a document file.

    Args:
        file: Uploaded file.
        patient_id: Optional patient ID (UUID string).
        encounter_id: Optional encounter ID (UUID string).
        upload_method: Upload method (camera, drag_and_drop, file_picker, etc.).
        session: Database session.
        file_service: File asset service.
        storage_service: Storage service.

    Returns:
        File upload response with file ID and metadata.
    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Filename is required"
            )

        # Convert string UUIDs to UUID objects if provided
        patient_uuid = None
        encounter_uuid = None

        if patient_id:
            try:
                patient_uuid = (
                    UUID(patient_id) if isinstance(patient_id, str) else patient_id
                )
            except (ValueError, TypeError):
                logger.warning("Invalid patient_id format: %s", patient_id)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid patient_id format: {patient_id}",
                ) from None

        if encounter_id:
            try:
                encounter_uuid = (
                    UUID(encounter_id)
                    if isinstance(encounter_id, str)
                    else encounter_id
                )
            except (ValueError, TypeError):
                logger.warning("Invalid encounter_id format: %s", encounter_id)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid encounter_id format: {encounter_id}",
                ) from None

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
        import sys

        print(
            f"[UPLOAD] Creating file asset: patient_id={patient_uuid}, encounter_id={encounter_uuid}, filename={file.filename}",  # noqa: E501
            file=sys.stderr,
            flush=True,
        )
        logger.info(
            "Creating file asset: patient_id=%s, encounter_id=%s, filename=%s",
            patient_uuid,
            encounter_uuid,
            file.filename,
        )

        asset = await file_service.create_file_asset(
            patient_id=patient_uuid,
            encounter_id=encounter_uuid,
            storage_path=str(storage_path),
            original_filename=file.filename,
            content_type=content_type,
            size_bytes=size_bytes,
            upload_method=upload_method or "file_picker",
            document_type=document_type,
        )

        print(
            f"[UPLOAD] File asset created: id={asset.id}, patient_id={asset.patient_id}",  # noqa: E501
            file=sys.stderr,
            flush=True,
        )
        logger.info(
            "File asset created: id=%s, patient_id=%s", asset.id, asset.patient_id
        )

        # Trigger document processing and summary update (if patient_id is provided)
        print(
            f"[UPLOAD] Checking if patient_id exists: {patient_uuid}",
            file=sys.stderr,
            flush=True,
        )
        if patient_uuid:
            print(
                "[UPLOAD] Patient ID found, triggering processing and summary update",
                file=sys.stderr,
                flush=True,
            )
            try:
                import httpx

                print(
                    "[UPLOAD] Starting document processing/summary trigger",
                    file=sys.stderr,
                    flush=True,
                )
                logger.info(
                    "Starting document processing/summary trigger for patient %s",
                    patient_uuid,
                )

                # Step 1: Process document to extract text (for PDFs and images)
                document_processed = False
                processing_result = None
                print(
                    f"[UPLOAD] Content type: {content_type}, checking if processable",
                    file=sys.stderr,
                    flush=True,
                )
                if content_type in [
                    "application/pdf",
                    "image/jpeg",
                    "image/png",
                    "image/tiff",
                ]:
                    try:
                        process_url = f"http://localhost:8003/summarizer/documents/{asset.id}/process"
                        print(
                            f"[UPLOAD] Calling document processor: {process_url}",
                            file=sys.stderr,
                            flush=True,
                        )
                        async with httpx.AsyncClient(
                            timeout=60.0
                        ) as client:  # Increased timeout for Gemini processing
                            process_response = await client.post(process_url)
                            print(
                                f"[UPLOAD] Document processing response: status={process_response.status_code}",  # noqa: E501
                                file=sys.stderr,
                                flush=True,
                            )
                            if process_response.status_code == 200:
                                processing_result = process_response.json()
                                success = processing_result.get("success", False)
                                has_extraction = bool(
                                    processing_result.get("overall_confidence", 0) > 0
                                )
                                logger.info(
                                    "Document %s processed: success=%s, confidence=%s",
                                    asset.id,
                                    success,
                                    processing_result.get("overall_confidence"),
                                )
                                print(
                                    f"[UPLOAD] Document {asset.id} processed: success={success}, confidence={processing_result.get('overall_confidence')}",  # noqa: E501
                                    file=sys.stderr,
                                    flush=True,
                                )
                                if success and has_extraction:
                                    document_processed = True
                                else:
                                    print(
                                        "[UPLOAD] WARNING: Document processing completed but extraction failed or low confidence",  # noqa: E501
                                        file=sys.stderr,
                                        flush=True,
                                    )
                                    logger.warning(
                                        "Document %s processing completed but extraction failed or low confidence",  # noqa: E501
                                        asset.id,
                                    )
                            else:
                                logger.warning(
                                    "Document processing returned status %d: %s",
                                    process_response.status_code,
                                    process_response.text,
                                )
                                print(
                                    f"[UPLOAD] Document processing returned status {process_response.status_code}: {process_response.text[:200]}",  # noqa: E501
                                    file=sys.stderr,
                                    flush=True,
                                )
                    except Exception as e:
                        logger.warning(
                            "Failed to process document (non-critical): %s",
                            e,
                            exc_info=True,
                        )
                        print(
                            f"[UPLOAD] Failed to process document: {e}",
                            file=sys.stderr,
                            flush=True,
                        )
                else:
                    print(
                        f"[UPLOAD] Content type {content_type} not processable, skipping document processing",  # noqa: E501
                        file=sys.stderr,
                        flush=True,
                    )

                # Step 2: Verify document processing completed and wait for DB commit
                if document_processed:
                    import asyncio

                    # Wait for database commit to propagate
                    print(
                        "[UPLOAD] Waiting 2 seconds for document processing to commit to DB",  # noqa: E501
                        file=sys.stderr,
                        flush=True,
                    )
                    await asyncio.sleep(2)

  #  Verify the file asset was updated with extraction_data (with retries)
                    max_retries = 5
                    has_extraction_data = False
                    for attempt in range(max_retries):
                        try:
                            # Use a fresh query to bypass any session caching
                            await session.commit()  # Commit any pending changes
                            from sqlalchemy import select

                            from database.models import FileAsset

                            stmt = select(FileAsset).where(FileAsset.id == asset.id)
                            result = await session.execute(stmt)
                            updated_asset = result.scalar_one_or_none()
                            if updated_asset:
                                has_extraction = bool(updated_asset.extraction_data)
                                has_raw_text = bool(updated_asset.raw_text)
                                print(
                                    f"[UPLOAD] Attempt {attempt+1}/{max_retries}: "
                                    f"File asset {asset.id}: "
                                    f"has_extraction_data={has_extraction}, "
                                    f"has_raw_text={has_raw_text}, "
                                    f"extraction_status={updated_asset.extraction_status}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                                if has_extraction:
                                    has_extraction_data = True
                                    logger.info(
                                        "File asset %s verified: has_extraction_data=True",  # noqa: E501
                                        asset.id,
                                    )
                                    break
                                else:
                                    if attempt < max_retries - 1:
                                        print(
                                            "[UPLOAD] Extraction data not yet available, waiting 1 more second...",  # noqa: E501
                                            file=sys.stderr,
                                            flush=True,
                                        )
                                        await asyncio.sleep(1)
                        except Exception as e:
                            print(
                                f"[UPLOAD] Error verifying file asset (attempt {attempt+1}): {e}",  # noqa: E501
                                file=sys.stderr,
                                flush=True,
                            )
                            if attempt < max_retries - 1:
                                await asyncio.sleep(1)

                    if not has_extraction_data:
                        print(
                            f"[UPLOAD] WARNING: File asset still has no extraction_data after {max_retries} attempts - proceeding anyway",  # noqa: E501
                            file=sys.stderr,
                            flush=True,
                        )
                        logger.warning(
                            "File asset %s still has no extraction_data after verification attempts",  # noqa: E501
                            asset.id,
                        )

                # Get all encounter IDs for this patient
                print(
                    f"[UPLOAD] Fetching encounter IDs for patient {patient_uuid}",
                    file=sys.stderr,
                    flush=True,
                )
                from sqlalchemy import select

                from database.models import Encounter

                stmt = select(Encounter.id).where(Encounter.patient_id == patient_uuid)
                result = await session.execute(stmt)
                encounter_ids = [str(eid) for eid in result.scalars().all()]
                print(
                    f"[UPLOAD] Found {len(encounter_ids)} encounters: {encounter_ids}",
                    file=sys.stderr,
                    flush=True,
                )

                # Call summarizer agent to update summary
                summarizer_url = "http://localhost:8003/summarizer/generate-summary"
                print(
                    f"[UPLOAD] Calling summarizer: {summarizer_url}",
                    file=sys.stderr,
                    flush=True,
                )
                async with httpx.AsyncClient(timeout=30.0) as client:
                    summary_response = await client.post(
                        summarizer_url,
                        json={
                            "patient_id": str(patient_uuid),
                            "encounter_ids": encounter_ids
                            if encounter_ids
                            else [str(encounter_uuid)]
                            if encounter_uuid
                            else [],
                            "highlights": [],
                        },
                    )
                    print(
                        f"[UPLOAD] Summary response: status={summary_response.status_code}",  # noqa: E501
                        file=sys.stderr,
                        flush=True,
                    )
                    if summary_response.status_code == 201:
                        logger.info(
                            "Summary updated successfully for patient %s", patient_uuid
                        )
                        print(
                            f"[UPLOAD] Summary updated successfully for patient {patient_uuid}",  # noqa: E501
                            file=sys.stderr,
                            flush=True,
                        )
                    else:
                        logger.warning(
                            "Summary update returned status %d: %s",
                            summary_response.status_code,
                            summary_response.text,
                        )
                        print(
                            f"[UPLOAD] Summary update returned status {summary_response.status_code}: {summary_response.text[:200]}",  # noqa: E501
                            file=sys.stderr,
                            flush=True,
                        )
            except Exception as e:
                import traceback

                logger.error(
                    "Failed to trigger document processing/summary update: %s",
                    e,
                    exc_info=True,
                )
                print(
                    f"[UPLOAD] EXCEPTION in trigger: {e}", file=sys.stderr, flush=True
                )
                print(traceback.format_exc(), file=sys.stderr, flush=True)
                # Don't fail the upload if processing fails
        else:
            print(
                "[UPLOAD] No patient_id provided, skipping document processing and summary update",  # noqa: E501
                file=sys.stderr,
                flush=True,
            )
            logger.warning(
                "No patient_id provided for file upload, skipping summary update"
            )

        return FileUploadResponse(
            file_id=asset.id,
            original_filename=asset.original_filename,
            size_bytes=asset.size_bytes,
            content_type=asset.content_type,
            status=asset.status,
            message="File uploaded successfully",
        )
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except ValueError as e:
        # UUID conversion errors
        logger.error("Invalid UUID format in upload: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid UUID format: {e!s}",
        ) from e
    except Exception as e:
        # All other errors
        import traceback

        logger.error("Document upload failed: %s", e, exc_info=True)
        print(f"[UPLOAD] FATAL ERROR: {e}", file=sys.stderr, flush=True)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {e!s}",
        ) from e


@router.get(
    "/documents/test-connection",
    summary="Test connection",
    description="Test if documents router is accessible",
)
async def test_documents_endpoint():
    """Test endpoint to verify documents router is accessible."""
    import sys

    print("[TEST] Documents endpoint called!", file=sys.stderr, flush=True)
    return {
        "status": "ok",
        "message": "Documents router is working",
        "endpoint": "/manage/documents/test-connection",
    }


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


@router.delete(
    "/documents/{file_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete document",
    description="Delete a document file and its metadata.",
    response_model=None,
)
async def delete_document(
    file_id: UUID,
    file_service: FileAssetService = Depends(get_file_asset_service),
) -> None:
    """
    Delete a document file and its metadata.

    Args:
        file_id: File asset ID.
        file_service: File asset service.
    """
    asset = await file_service.get_file_asset(file_id)
    if asset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )
    await file_service.delete_file_asset(asset)
