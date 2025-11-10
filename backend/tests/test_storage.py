import importlib
import tempfile
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import UploadFile

from backend.services import config as config_module


@pytest.mark.asyncio
async def test_local_storage_saves_and_resolves(monkeypatch):
    with tempfile.TemporaryDirectory() as storage_dir, tempfile.TemporaryDirectory() as db_dir:
        db_path = Path(db_dir) / "test.db"

        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
        monkeypatch.setenv("STORAGE_BACKEND", "local")
        monkeypatch.setenv("STORAGE_LOCAL_PATH", storage_dir)
        monkeypatch.delenv("STORAGE_GCS_BUCKET", raising=False)

        config_module.get_settings.cache_clear()

        session_module = importlib.import_module("backend.database.session")
        importlib.reload(session_module)
        importlib.import_module("backend.database.models")
        base_module = importlib.import_module("backend.database.base")
        base_module.Base.metadata.create_all(bind=session_module.engine)

        storage_module = importlib.import_module("backend.services.storage")
        importlib.reload(storage_module)
        storage_service = storage_module.StorageService()

        upload = UploadFile(
            filename="test.wav",
            file=BytesIO(b"fake audio"),
            headers={"content-type": "audio/wav"},
        )
        record, stored = await storage_service.save_audio_file(upload)

        try:
            assert record.id == stored.file_id
            assert stored.size_bytes == len(b"fake audio")
            assert record.storage_path == stored.storage_path

            resolved_path = storage_service.resolve_file_path(record)
            assert Path(resolved_path).exists()

            fetched = storage_service.get_audio_file(record.id)
            assert fetched is not None
            assert fetched.id == record.id
        finally:
            session_module.engine.dispose()

