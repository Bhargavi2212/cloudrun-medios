"""Storage abstraction supporting local filesystem and Google Cloud Storage."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional, Protocol
from uuid import uuid4

from fastapi import UploadFile

from sqlalchemy import select

from ..database import crud
from ..database.models import AudioFile, DocumentProcessingStatus, FileAsset
from ..database.session import get_session
from .config import get_settings

logger = logging.getLogger(__name__)

try:
    from google.cloud import storage as gcs_storage  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    gcs_storage = None


class StorageError(Exception):
    """Raised when storage operations fail."""


@dataclass
class StoredFile:
    file_id: str
    storage_path: str
    size_bytes: int
    content_type: Optional[str]
    checksum: str
    bucket: Optional[str] = None
    signed_url: Optional[str] = None


class StorageProvider(Protocol):
    async def store_file(self, upload_file: UploadFile) -> StoredFile: ...

    def resolve_path(self, storage_path: str) -> str: ...

    def delete_file(self, storage_path: str) -> None: ...

    def generate_signed_url(self, storage_path: str, *, expires_in: int) -> Optional[str]: ...


class LocalStorageProvider(StorageProvider):
    def __init__(self, base_path: Path) -> None:
        # Ensure base_path is absolute
        self.base_path = Path(base_path).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def store_file(self, upload_file: UploadFile) -> StoredFile:
        file_id = str(uuid4())
        suffix = Path(upload_file.filename or "").suffix or ".bin"
        relative_path = Path(file_id[0:2]) / f"{file_id}{suffix}"
        target_path = self.base_path / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        hasher = hashlib.sha256()
        size = 0
        with target_path.open("wb") as outfile:
            while True:
                chunk = await upload_file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                hasher.update(chunk)
                outfile.write(chunk)
        checksum = hasher.hexdigest()
        return StoredFile(
            file_id=file_id,
            storage_path=str(relative_path).replace("\\", "/"),
            size_bytes=size,
            content_type=upload_file.content_type,
            checksum=checksum,
            bucket=None,
            signed_url=None,
        )

    def resolve_path(self, storage_path: str) -> str:
        # Ensure we return an absolute path
        # base_path is already absolute, so just join and normalize
        resolved = (self.base_path / storage_path.replace("\\", "/")).resolve()
        # Double-check it's absolute
        if not resolved.is_absolute():
            # Fallback: use absolute path from current working directory
            resolved = Path.cwd() / resolved
            resolved = resolved.resolve()
        return str(resolved)

    def delete_file(self, storage_path: str) -> None:
        try:
            (self.base_path / storage_path).unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover - best effort cleanup
            raise StorageError(str(exc)) from exc

    def generate_signed_url(self, storage_path: str, *, expires_in: int) -> Optional[str]:
        # Local backend does not issue signed URLs; clients can proxy via API.
        return None


class GCSStorageProvider(StorageProvider):
    def __init__(self, bucket_name: str, credentials_path: Optional[Path], signed_url_expiry: int) -> None:
        if gcs_storage is None:  # pragma: no cover - optional dependency
            raise StorageError("google-cloud-storage is not installed")

        if credentials_path:
            self.client = gcs_storage.Client.from_service_account_json(str(credentials_path))
        else:
            self.client = gcs_storage.Client()

        self.bucket_name = bucket_name
        self.bucket = self.client.bucket(bucket_name)
        if not self.bucket:
            raise StorageError(f"GCS bucket '{bucket_name}' not found")
        self._default_signed_url_expiry = signed_url_expiry

    async def store_file(self, upload_file: UploadFile) -> StoredFile:
        file_id = str(uuid4())
        suffix = Path(upload_file.filename or "").suffix or ".bin"
        blob_name = f"audio/{file_id}{suffix}"
        blob = self.bucket.blob(blob_name)

        hasher = hashlib.sha256()
        buffer = BytesIO()
        size = 0
        while True:
            chunk = await upload_file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            hasher.update(chunk)
            buffer.write(chunk)
        buffer.seek(0)

        await asyncio.to_thread(
            blob.upload_from_file,
            buffer,
            content_type=upload_file.content_type or "application/octet-stream",
            rewind=True,
        )

        signed_url = self.generate_signed_url(blob_name, expires_in=self._default_signed_url_expiry)
        return StoredFile(
            file_id=file_id,
            storage_path=blob_name,
            size_bytes=size,
            content_type=upload_file.content_type,
            checksum=hasher.hexdigest(),
            bucket=self.bucket_name,
            signed_url=signed_url,
        )

    def resolve_path(self, storage_path: str) -> str:
        blob = self.bucket.blob(storage_path)
        if not blob.exists():
            raise StorageError(f"GCS object '{storage_path}' not found")
        tmp_dir = Path(tempfile.gettempdir()) / "medios-gcs"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        local_path = tmp_dir / f"{uuid4()}-{Path(storage_path).name}"
        blob.download_to_filename(str(local_path))
        return str(local_path)

    def delete_file(self, storage_path: str) -> None:
        blob = self.bucket.blob(storage_path)
        if blob.exists():
            blob.delete()

    def generate_signed_url(self, storage_path: str, *, expires_in: int) -> Optional[str]:
        blob = self.bucket.blob(storage_path)
        if not blob.exists():
            return None
        return blob.generate_signed_url(expiration=expires_in)


class StorageService:
    def __init__(self, provider: Optional[StorageProvider] = None) -> None:
        self.settings = get_settings()
        self.provider = provider or self._build_provider()
        self._is_local_backend = isinstance(self.provider, LocalStorageProvider)

    def _build_provider(self) -> StorageProvider:
        backend = self.settings.storage_backend.lower()
        if backend == "local":
            return LocalStorageProvider(self.settings.storage_local_path)
        if backend == "gcs":
            if not self.settings.storage_gcs_bucket:
                raise StorageError("STORAGE_GCS_BUCKET is required when using GCS backend")
            return GCSStorageProvider(
                self.settings.storage_gcs_bucket,
                self.settings.storage_gcs_credentials,
                self.settings.storage_signed_url_expiry_seconds,
            )
        raise StorageError(f"Unsupported storage backend: {backend}")

    async def save_audio_file(
        self,
        upload_file: UploadFile,
        *,
        consultation_id: Optional[str] = None,
        uploaded_by: Optional[str] = None,
    ) -> tuple[AudioFile, StoredFile]:
        stored_file = await self.provider.store_file(upload_file)
        signed_url = stored_file.signed_url or self.provider.generate_signed_url(
            stored_file.storage_path,
            expires_in=self.settings.storage_signed_url_expiry_seconds,
        )

        with get_session() as session:
            record = crud.create_audio_file(
                session,
                id=stored_file.file_id,
                consultation_id=consultation_id,
                uploaded_by=uploaded_by,
                original_filename=upload_file.filename,
                storage_path=stored_file.storage_path,
                duration_seconds=None,
                mime_type=stored_file.content_type,
                size_bytes=stored_file.size_bytes,
                checksum=stored_file.checksum,
                status="uploaded",
            )
            session.refresh(record)

        stored_file.signed_url = signed_url
        return record, stored_file

    def get_audio_file(self, audio_id: str) -> Optional[AudioFile]:
        with get_session() as session:
            return crud.get_audio_file(session, audio_id)

    def get_file_asset(self, file_id: str) -> Optional[FileAsset]:
        with get_session() as session:
            return crud.get_file_asset(session, file_id)

    def resolve_file_path(self, audio_file: AudioFile) -> str:
        return self.provider.resolve_path(audio_file.storage_path)

    def generate_signed_url(self, audio_file: AudioFile) -> Optional[str]:
        return self.provider.generate_signed_url(
            audio_file.storage_path,
            expires_in=self.settings.storage_signed_url_expiry_seconds,
        )

    def delete_audio_file(self, audio_file: AudioFile) -> None:
        self.provider.delete_file(audio_file.storage_path)
        with get_session() as session:
            crud.soft_delete(session, audio_file)

    def _get_retention_policy(self, file_type: Optional[str] = None) -> Optional[str]:
        """Get retention policy string based on file type.
        
        Args:
            file_type: Optional file type hint ("audio", "document", etc.)
            
        Returns:
            Retention policy string (e.g., "365d", "2555d") or None if disabled
        """
        if not self.settings.storage_retention_enabled:
            return None
        
        if file_type == "audio":
            return f"{self.settings.storage_retention_audio_days}d"
        elif file_type == "document":
            return f"{self.settings.storage_retention_document_days}d"
        else:
            return f"{self.settings.storage_retention_default_days}d"
    
    def _parse_retention_days(self, retention_policy: Optional[str]) -> Optional[int]:
        """Parse retention policy string to days.
        
        Args:
            retention_policy: Retention policy string (e.g., "365d", "2555d")
            
        Returns:
            Number of days or None if invalid
        """
        if not retention_policy:
            return None
        try:
            # Remove 'd' suffix and parse as int
            days_str = retention_policy.rstrip("d").strip()
            return int(days_str)
        except (ValueError, AttributeError):
            logger.warning(f"Invalid retention policy format: {retention_policy}")
            return None
    
    async def save_file_asset(
        self,
        upload_file: UploadFile,
        *,
        owner_type: str,
        owner_id: str,
        uploaded_by: Optional[str] = None,
        description: Optional[str] = None,
        file_type: Optional[str] = None,
        retention_policy: Optional[str] = None,
    ) -> tuple[FileAsset, StoredFile]:
        """Save file asset with retention policy.
        
        Args:
            upload_file: File to upload
            owner_type: Owner type (e.g., "patient", "consultation")
            owner_id: Owner ID
            uploaded_by: User ID who uploaded the file
            description: Optional file description
            file_type: Optional file type hint for automatic retention policy ("audio", "document")
            retention_policy: Optional explicit retention policy (e.g., "365d"). If not provided, determined from file_type
        """
        stored_file = await self.provider.store_file(upload_file)
        signed_url = stored_file.signed_url or self.provider.generate_signed_url(
            stored_file.storage_path,
            expires_in=self.settings.storage_signed_url_expiry_seconds,
        )
        
        # Determine retention policy
        if retention_policy is None:
            retention_policy = self._get_retention_policy(file_type)

        with get_session() as session:
            record = crud.create_file_asset(
                session,
                id=stored_file.file_id,
                owner_type=owner_type,
                owner_id=owner_id,
                original_filename=upload_file.filename,
                storage_path=stored_file.storage_path,
                bucket=stored_file.bucket,
                content_type=stored_file.content_type,
                size_bytes=stored_file.size_bytes,
                checksum=stored_file.checksum,
                retention_policy=retention_policy,
                uploaded_by=uploaded_by,
                description=description,
                status=DocumentProcessingStatus.UPLOADED,
            )
            session.refresh(record)

        stored_file.signed_url = signed_url
        return record, stored_file

    def resolve_file_asset_path(self, asset: FileAsset) -> str:
        return self.provider.resolve_path(asset.storage_path)

    def generate_file_asset_signed_url(self, asset: FileAsset) -> Optional[str]:
        return self.provider.generate_signed_url(
            asset.storage_path,
            expires_in=self.settings.storage_signed_url_expiry_seconds,
        )

    def delete_file_asset(self, asset: FileAsset) -> None:
        self.provider.delete_file(asset.storage_path)
        with get_session() as session:
            crud.soft_delete(session, asset)

    def get_files_for_cleanup(self, batch_size: int = 100) -> list[FileAsset]:
        """Get files that have exceeded their retention policy.
        
        Args:
            batch_size: Maximum number of files to return
            
        Returns:
            List of FileAsset objects that should be deleted
        """
        if not self.settings.storage_retention_enabled:
            return []
        
        cutoff_date = datetime.now(timezone.utc)
        expired_files = []
        
        with get_session() as session:
            # Query all non-deleted file assets with retention policies
            stmt = (
                select(FileAsset)
                .where(
                    FileAsset.is_deleted.is_(False),
                    FileAsset.retention_policy.isnot(None),
                )
                .order_by(FileAsset.created_at.asc())  # Oldest first
                .limit(batch_size * 2)  # Get more to filter
            )
            files = session.execute(stmt).scalars().all()
            
            for file_asset in files:
                # Parse retention policy
                retention_days = self._parse_retention_days(file_asset.retention_policy)
                if retention_days is None:
                    # Invalid retention policy, skip
                    continue
                
                # Calculate expiry date
                expiry_date = file_asset.created_at + timedelta(days=retention_days)
                
                # Check if expired
                if expiry_date < cutoff_date:
                    expired_files.append(file_asset)
                    if len(expired_files) >= batch_size:
                        break
        
        return expired_files
    
    def cleanup_expired_files(self, batch_size: int = 100) -> dict[str, int]:
        """Clean up files that have exceeded their retention policy.
        
        Args:
            batch_size: Maximum number of files to process in one run
            
        Returns:
            Dictionary with cleanup statistics: {"deleted": count, "errors": count}
        """
        if not self.settings.storage_retention_enabled:
            logger.info("Storage retention cleanup is disabled")
            return {"deleted": 0, "errors": 0}
        
        expired_files = self.get_files_for_cleanup(batch_size=batch_size)
        deleted_count = 0
        error_count = 0
        
        logger.info(f"Found {len(expired_files)} files exceeding retention policy")
        
        for file_asset in expired_files:
            try:
                # Delete the physical file
                self.provider.delete_file(file_asset.storage_path)
                
                # Soft delete the database record
                with get_session() as session:
                    crud.soft_delete(session, file_asset)
                
                deleted_count += 1
                logger.debug(f"Deleted expired file: {file_asset.id} ({file_asset.original_filename})")
            except Exception as exc:
                error_count += 1
                logger.error(f"Failed to delete expired file {file_asset.id}: {exc}")
        
        logger.info(f"Retention cleanup completed: {deleted_count} deleted, {error_count} errors")
        return {"deleted": deleted_count, "errors": error_count}
    
    @property
    def is_local_backend(self) -> bool:
        return self._is_local_backend

