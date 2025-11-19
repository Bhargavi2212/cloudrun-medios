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
import contextlib
import mimetypes
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy import select

from ..database import crud
from ..database.models import AudioFile, DocumentProcessingStatus, FileAsset
from ..database.session import get_session
from .config import get_settings

logger = logging.getLogger(__name__)

IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/heic",
    "image/heif",
    "image/tiff",
}
ALLOWED_DOCUMENT_MIME_TYPES = {"application/pdf"} | IMAGE_MIME_TYPES
DEFAULT_UPLOAD_METHOD = "file_picker"
MAX_RAW_TEXT_FALLBACK = 50000

try:
    from PyPDF2 import PdfReader  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    PdfReader = None

try:
    from google.cloud import storage as gcs_storage  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    gcs_storage = None

try:
    from google.cloud import vision  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    vision = None


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
        self._vision_client = None
        self._vision_enabled = bool(self.settings.vision_api_enabled and vision is not None)
        if self.settings.vision_api_enabled and vision is None:
            logger.warning(
                "VISION_API_ENABLED is true but google-cloud-vision is not installed; OCR will be limited to PDF extraction."
            )

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

    def _get_vision_client(self):
        if not self._vision_enabled:
            return None
        if self._vision_client is None:
            if self.settings.vision_credentials_path:
                self._vision_client = vision.ImageAnnotatorClient.from_service_account_json(
                    str(self.settings.vision_credentials_path)
                )
            else:
                self._vision_client = vision.ImageAnnotatorClient()
        return self._vision_client

    def _infer_document_type(self, filename: Optional[str], content_type: Optional[str]) -> Optional[str]:
        if (content_type and content_type.lower() == "application/pdf") or (
            filename and filename.lower().endswith(".pdf")
        ):
            return "pdf"
        if content_type and content_type.lower() in IMAGE_MIME_TYPES:
            return "image"
        return None

    def _truncate_text(self, text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        limit = max(1000, self.settings.document_raw_text_max_chars or MAX_RAW_TEXT_FALLBACK)
        if len(text) <= limit:
            return text
        return text[:limit]

    def _score_extraction_confidence(self, text: Optional[str]) -> Tuple[Optional[float], Optional[str]]:
        if not text:
            return None, "red"
        length = len(text)
        min_chars = max(1, self.settings.document_min_text_chars)
        if length >= min_chars * 5:
            return 0.98, "green"
        if length >= min_chars * 2:
            return 0.92, "yellow"
        return 0.75, "yellow"

    def _validate_file_size(self, stored_file: StoredFile) -> None:
        max_bytes = max(1, self.settings.document_upload_max_mb) * 1024 * 1024
        if stored_file.size_bytes > max_bytes:
            self.provider.delete_file(stored_file.storage_path)
            raise StorageError(
                f"File '{stored_file.file_id}' exceeds maximum allowed size of {self.settings.document_upload_max_mb} MB."
            )

    @staticmethod
    def _extract_pdf_text(path: str) -> Tuple[Optional[str], Dict[str, Any]]:
        if PdfReader is None:
            return None, {"source": "pypdf2", "error": "PyPDF2 not installed"}
        try:
            reader = PdfReader(path)
            pages = []
            for page in reader.pages:
                try:
                    pages.append(page.extract_text() or "")
                except Exception:  # pragma: no cover - PDF quirks
                    pages.append("")
            text = "\n".join(fragment for fragment in pages if fragment).strip()
            return text or None, {"source": "pypdf2", "page_count": len(reader.pages)}
        except Exception as exc:  # pragma: no cover - PDF parsing errors
            return None, {"source": "pypdf2", "error": str(exc)}

    async def _extract_image_text(self, path: str, content_type: str) -> Tuple[Optional[str], Dict[str, Any]]:
        client = self._get_vision_client()
        if client is None:
            return None, {"source": "vision", "error": "disabled"}

        def _run_detection():
            with open(path, "rb") as image_file:
                image = vision.Image(content=image_file.read())
            response = client.document_text_detection(image=image)
            if response.error.message:
                raise RuntimeError(response.error.message)
            return response

        try:
            response = await asyncio.to_thread(_run_detection)
        except Exception as exc:  # pragma: no cover - Vision API errors
            logger.warning("Vision OCR failed for %s: %s", path, exc)
            return None, {"source": "vision", "error": str(exc)}

        annotation = response.full_text_annotation
        text = annotation.text.strip() if annotation and annotation.text else None
        languages = []
        if annotation and annotation.pages:
            for page in annotation.pages:
                if not page.property:
                    continue
                for lang in page.property.detected_languages:
                    if lang.language_code:
                        languages.append(lang.language_code)
        metadata: Dict[str, Any] = {
            "source": "vision_document",
            "languages": list(dict.fromkeys(languages)),
            "content_type": content_type,
        }
        return text or None, metadata

    async def _extract_text_preview(
        self,
        stored_file: StoredFile,
        *,
        filename: Optional[str],
    ) -> Tuple[Optional[str], Dict[str, Any], bool]:
        content_type = stored_file.content_type or mimetypes.guess_type(filename or "")[0] or "application/octet-stream"
        resolved_path = self.provider.resolve_path(stored_file.storage_path)
        cleanup_after_use = not self._is_local_backend
        text: Optional[str] = None
        metadata: Dict[str, Any] = {"content_type": content_type}
        try:
            if content_type == "application/pdf":
                text, pdf_meta = await asyncio.to_thread(self._extract_pdf_text, resolved_path)
                metadata.update(pdf_meta or {})
            elif content_type in IMAGE_MIME_TYPES:
                text, image_meta = await self._extract_image_text(resolved_path, content_type)
                metadata.update(image_meta or {})
            else:
                metadata["reason"] = "unsupported_content_type"
            truncated = self._truncate_text(text)
            needs_review = not bool(truncated and len(truncated) >= self.settings.document_min_text_chars)
            metadata["char_count"] = len(truncated or "") if truncated else 0
            return truncated, metadata, needs_review
        finally:
            if cleanup_after_use:
                with contextlib.suppress(OSError):
                    Path(resolved_path).unlink(missing_ok=True)

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
        document_type: Optional[str] = None,
        upload_method: Optional[str] = None,
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
        self._validate_file_size(stored_file)
        upload_method = upload_method or DEFAULT_UPLOAD_METHOD
        signed_url = stored_file.signed_url or self.provider.generate_signed_url(
            stored_file.storage_path,
            expires_in=self.settings.storage_signed_url_expiry_seconds,
        )

        # Determine retention policy
        if retention_policy is None:
            retention_policy = self._get_retention_policy(file_type)

        inferred_document_type = document_type or self._infer_document_type(
            upload_file.filename, stored_file.content_type
        )
        raw_text: Optional[str] = None
        extraction_metadata: Dict[str, Any] = {}
        needs_review = True
        try:
            raw_text, extraction_metadata, needs_review = await self._extract_text_preview(
                stored_file,
                filename=upload_file.filename,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Initial OCR preview failed for %s: %s", stored_file.file_id, exc)
            extraction_metadata = {"error": str(exc)}
            needs_review = True
        extraction_confidence, confidence_tier = self._score_extraction_confidence(raw_text)
        extraction_status = "extracted" if raw_text else "pending_ocr"
        review_status = "pending" if needs_review else "auto_approved"
        extraction_payload = {
            **(extraction_metadata or {}),
            "extraction_status": extraction_status,
            "review_status": review_status,
        }

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
                document_type=inferred_document_type,
                upload_method=upload_method,
                raw_text=raw_text,
                extraction_status=extraction_status,
                extraction_confidence=extraction_confidence,
                confidence_tier=confidence_tier,
                review_status=review_status,
                needs_manual_review=needs_review,
                extraction_data=extraction_payload,
                processing_metadata={"ocr": extraction_payload},
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
