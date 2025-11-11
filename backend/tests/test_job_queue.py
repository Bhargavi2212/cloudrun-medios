import asyncio
from io import BytesIO
from pathlib import Path

import pytest

from backend.services import config as config_module
from backend.services.error_response import StandardResponse
from backend.services.job_queue import JobQueueService
from backend.services.make_agent import MakeAgentService
from backend.services.storage import StorageService


class FakeMakeAgentService(MakeAgentService):
    async def process_audio_pipeline(self, audio_file_path: str, *, consultation_id=None, author_id=None) -> StandardResponse:
        return StandardResponse(
            success=True,
            data={
                "transcription": "hello",
                "entities": {
                    "symptoms": [],
                    "medications": [],
                    "diagnoses": [],
                    "vitals": {},
                },
                "generated_note": "S: ...",
                "confidence_scores": {},
                "stage_completed": "completed",
                "warnings": [],
                "note_version_id": None,
            },
            is_stub=False,
        )


@pytest.mark.asyncio
async def test_enqueue_and_process_job(tmp_path, monkeypatch):
    db_path = tmp_path / "job.db"
    storage_dir = tmp_path / "storage"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "uploads"))
    monkeypatch.setenv("WHISPER_CACHE_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("STORAGE_LOCAL_PATH", str(storage_dir))

    config_module.get_settings.cache_clear()

    # Reload modules that depend on settings
    session_module = __import__("backend.database.session", fromlist=["engine"])
    import importlib

    importlib.reload(session_module)
    importlib.import_module("backend.database.models")
    base_module = importlib.import_module("backend.database.base")
    base_module.Base.metadata.create_all(bind=session_module.engine)

    storage_service = StorageService()
    upload = UploadFileMock("sample.wav", b"fake audio data", "audio/wav")
    record, _ = await storage_service.save_audio_file(upload, consultation_id="consult-123")

    job_service = JobQueueService(
        make_agent_service=FakeMakeAgentService(),
        storage_service=storage_service,
    )

    job = await job_service.enqueue_audio_processing(record.id)
    await asyncio.sleep(0.1)
    updated = job_service.get_job(job.id)

    assert updated is not None
    assert updated.status == "completed"
    assert updated.payload.get("result") is not None

    session_module.engine.dispose()


class UploadFileMock:
    def __init__(self, filename: str, data: bytes, content_type: str) -> None:
        self.filename = filename
        self.file = BytesIO(data)
        self.content_type = content_type

    async def read(self, size: int = -1) -> bytes:
        return self.file.read(size)
