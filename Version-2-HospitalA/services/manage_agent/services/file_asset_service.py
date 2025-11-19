"""
Business logic for file asset management.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.crud import AsyncCRUDRepository
from database.models import FileAsset
from services.manage_agent.services.storage_service import StorageService


class FileAssetService:
    """
    Coordinates file asset persistence and storage operations.
    """

    def __init__(self, session: AsyncSession, storage_service: StorageService) -> None:
        """
        Initialize file asset service.

        Args:
            session: Database session.
            storage_service: Storage service for file operations.
        """
        self.session = session
        self.storage_service = storage_service
        self.repository = AsyncCRUDRepository[FileAsset](session, FileAsset)

    async def create_file_asset(
        self,
        *,
        patient_id: UUID | None = None,
        encounter_id: UUID | None = None,
        storage_path: str,
        original_filename: str | None = None,
        content_type: str | None = None,
        size_bytes: int | None = None,
        upload_method: str | None = None,
        document_type: str | None = None,
    ) -> FileAsset:
        """
        Create a new file asset record.

        Args:
            patient_id: Optional patient ID.
            encounter_id: Optional encounter ID.
            storage_path: Path to stored file (relative to storage root).
            original_filename: Original filename.
            content_type: MIME type.
            size_bytes: File size in bytes.
            upload_method: Upload method (camera, drag_and_drop, etc.).
            document_type: Document type (lab_report, discharge, etc.).

        Returns:
            Created FileAsset instance.
        """
        payload = {
            "patient_id": patient_id,
            "encounter_id": encounter_id,
            "storage_path": str(storage_path),
            "original_filename": original_filename,
            "content_type": content_type,
            "size_bytes": size_bytes,
            "upload_method": upload_method,
            "document_type": document_type,
            "status": "uploaded",
        }
        asset = await self.repository.create(payload)
        await self.session.commit()
        await self.session.refresh(asset)
        return asset

    async def get_file_asset(self, file_id: UUID) -> FileAsset | None:
        """Retrieve a file asset by ID."""
        return await self.repository.get(file_id)

    async def list_file_assets(
        self,
        *,
        patient_id: UUID | None = None,
        encounter_id: UUID | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[FileAsset]:
        """
        List file assets with optional filters.

        Args:
            patient_id: Filter by patient ID.
            encounter_id: Filter by encounter ID.
            status: Filter by status.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of FileAsset instances.
        """
        stmt = select(FileAsset)
        if patient_id:
            stmt = stmt.where(FileAsset.patient_id == patient_id)
        if encounter_id:
            stmt = stmt.where(FileAsset.encounter_id == encounter_id)
        if status:
            stmt = stmt.where(FileAsset.status == status)
        stmt = stmt.offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def update_file_asset(
        self,
        asset: FileAsset,
        *,
        status: str | None = None,
        extraction_status: str | None = None,
        extraction_confidence: float | None = None,
        confidence_tier: str | None = None,
        review_status: str | None = None,
        needs_manual_review: bool | None = None,
        extraction_data: dict | None = None,
        processing_metadata: dict | None = None,
        processing_notes: str | None = None,
        last_error: str | None = None,
        raw_text: str | None = None,
    ) -> FileAsset:
        """
        Update a file asset record.

        Args:
            asset: FileAsset instance to update.
            **kwargs: Fields to update.

        Returns:
            Updated FileAsset instance.
        """
        update_data = {}
        if status is not None:
            update_data["status"] = status
        if extraction_status is not None:
            update_data["extraction_status"] = extraction_status
        if extraction_confidence is not None:
            update_data["extraction_confidence"] = extraction_confidence
        if confidence_tier is not None:
            update_data["confidence_tier"] = confidence_tier
        if review_status is not None:
            update_data["review_status"] = review_status
        if needs_manual_review is not None:
            update_data["needs_manual_review"] = needs_manual_review
        if extraction_data is not None:
            update_data["extraction_data"] = extraction_data
        if processing_metadata is not None:
            update_data["processing_metadata"] = processing_metadata
        if processing_notes is not None:
            update_data["processing_notes"] = processing_notes
        if last_error is not None:
            update_data["last_error"] = last_error
        if raw_text is not None:
            update_data["raw_text"] = raw_text

        updated = await self.repository.update(asset, update_data)
        await self.session.commit()
        await self.session.refresh(updated)
        return updated

    async def delete_file_asset(self, asset: FileAsset) -> None:
        """
        Delete a file asset and its associated file.

        Args:
            asset: FileAsset instance to delete.
        """
        # Delete file from storage
        await self.storage_service.delete_file(asset.storage_path)
        # Delete database record
        await self.repository.delete(asset)
        await self.session.commit()
