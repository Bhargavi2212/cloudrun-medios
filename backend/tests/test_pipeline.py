import asyncio
from pathlib import Path

import pytest

from backend.services.ai_models import AIModelsService
from backend.services.make_agent_pipeline import MakeAgentPipeline


class StubModels(AIModelsService):
    def __init__(self):
        # Do not call parent initialiser to avoid Gemini setup
        pass

    async def transcribe_audio(self, path: str):
        return {"success": True, "transcription": "Patient has chest pain.", "confidence": 0.9, "is_stub": False}

    async def extract_entities(self, text: str):
        return {
            "success": True,
            "entities": {"symptoms": ["chest pain"], "medications": [], "diagnoses": [], "vitals": {}},
            "confidence": 0.6,
            "is_stub": False,
        }

    async def generate_note(self, transcription, entities):
        return {"success": True, "generated_note": "S: chest pain.\nO: ...", "confidence": 0.8, "is_stub": False}


@pytest.mark.asyncio
async def test_pipeline_completes_successfully(tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"fake audio")

    pipeline = MakeAgentPipeline(models_service=StubModels())
    result = await pipeline.process_audio(str(audio_file))

    assert result["stage_completed"] == "completed"
    assert result["generated_note"].startswith("S:")

