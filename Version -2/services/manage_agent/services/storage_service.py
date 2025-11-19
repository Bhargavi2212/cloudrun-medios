"""
File storage service for document uploads.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import UploadFile

logger = logging.getLogger(__name__)


class StorageService:
    """
    Handles file storage operations (local filesystem for now).
    """

    def __init__(self, storage_root: Path | str) -> None:
        """
        Initialize storage service.

        Args:
            storage_root: Root directory for storing uploaded files.
        """
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        logger.info("Storage service initialized with root: %s", self.storage_root)

    def _get_file_hash(self, content: bytes) -> str:
        """Generate SHA256 hash of file content."""
        return hashlib.sha256(content).hexdigest()

    def _get_storage_path(self, file_id: UUID, filename: str) -> Path:
        """
        Generate storage path for a file using subdirectory structure.

        Uses first 2 characters of UUID for directory structure.
        """
        file_id_str = str(file_id)
        subdir = self.storage_root / file_id_str[:2]
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / file_id_str

    async def save_file(
        self,
        upload_file: UploadFile,
        *,
        file_id: UUID | None = None,
    ) -> tuple[UUID, Path, int, str]:
        """
        Save an uploaded file to storage.

        Args:
            upload_file: FastAPI UploadFile object.
            file_id: Optional file ID (generates new UUID if not provided).

        Returns:
            Tuple of (file_id, storage_path, size_bytes, content_hash).
        """
        if file_id is None:
            file_id = uuid4()

        # Read file content
        content = await upload_file.read()
        size_bytes = len(content)
        content_hash = self._get_file_hash(content)

        # Determine storage path
        storage_path = self._get_storage_path(
            file_id, upload_file.filename or "unknown"
        )

        # Save file
        storage_path.write_bytes(content)
        logger.info(
            "Saved file: id=%s, path=%s, size=%d bytes, hash=%s",
            file_id,
            storage_path,
            size_bytes,
            content_hash[:16],
        )

        # Return relative path for database storage
        relative_path = storage_path.relative_to(self.storage_root)
        return file_id, relative_path, size_bytes, content_hash

    def get_file_path(self, storage_path: str | Path) -> Path:
        """
        Get absolute path to a stored file.

        Args:
            storage_path: Relative storage path (as stored in database).

        Returns:
            Absolute path to the file.
        """
        if isinstance(storage_path, str):
            return self.storage_root / storage_path
        return self.storage_root / storage_path

    async def delete_file(self, storage_path: str | Path) -> None:
        """
        Delete a stored file.

        Args:
            storage_path: Relative storage path (as stored in database).
        """
        file_path = self.get_file_path(storage_path)
        if file_path.exists():
            file_path.unlink()
            logger.info("Deleted file: %s", file_path)

    def file_exists(self, storage_path: str | Path) -> bool:
        """Check if a file exists in storage."""
        return self.get_file_path(storage_path).exists()
