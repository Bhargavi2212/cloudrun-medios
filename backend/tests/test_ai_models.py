import asyncio
import shutil
from pathlib import Path

import pytest

from backend.services import ai_models


@pytest.fixture(autouse=True)
def reset_settings(monkeypatch):
    ai_models.get_settings.cache_clear()
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("DATA_PATH", str(Path.cwd() / "uploads"))
    monkeypatch.setenv("WHISPER_CACHE_DIR", str(Path.cwd() / "models"))
    monkeypatch.setenv("MODEL_SIZE", "tiny")


@pytest.mark.asyncio
async def test_transcribe_audio_success(tmp_path, monkeypatch):
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"\x00" * 4096)

    class FakeWhisper:
        def transcribe(self, *_args, **_kwargs):
            return {"text": "Hello world", "segments": [{"avg_logprob": -0.1}]}

    # Mock ffmpeg availability check to return True
    monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/ffmpeg" if cmd == "ffmpeg" else None)
    monkeypatch.setattr(ai_models, "get_whisper_model", lambda: FakeWhisper())

    service = ai_models.AIModelsService()
    result = await service.transcribe_audio(str(audio_file))

    assert result["success"] is True
    assert result["transcription"] == "Hello world"
    assert result["is_stub"] is False


@pytest.mark.asyncio
async def test_transcribe_audio_error_when_missing(tmp_path):
    service = ai_models.AIModelsService()
    result = await service.transcribe_audio(str(tmp_path / "missing.wav"))
    assert result["success"] is False
    assert "not found" in result["error"]


@pytest.mark.asyncio
async def test_extract_entities_identifies_keywords():
    service = ai_models.AIModelsService()
    text = "Patient reports chest pain and fever. BP is 160/95."
    result = await service.extract_entities(text)
    entities = result["entities"]
    assert "chest pain" in entities["symptoms"]
    assert entities["vitals"]["blood_pressure"] == "160/95"


@pytest.mark.asyncio
async def test_generate_note_uses_template_without_gemini(monkeypatch):
    service = ai_models.AIModelsService()
    result = await service.generate_note(
        "Patient has cough.", {"symptoms": ["cough"], "vitals": {}}
    )
    assert result["success"] is True
    assert result["is_stub"] is True
    assert "cough" in result["generated_note"].lower()
