"""
High-level facade for the AI Scribe pipeline.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, Optional

from backend.database import crud
from backend.database.session import get_session

from .ai_models import AIModelsService
from .error_response import StandardResponse
from .make_agent_pipeline import MakeAgentPipeline
from .telemetry import record_llm_usage, record_service_metric


class MakeAgentService:
    """Entrypoint used by FastAPI routes and other callers."""

    def __init__(self, models_service: Optional[AIModelsService] = None) -> None:
        self.ai_models = models_service or AIModelsService()
        self.pipeline = MakeAgentPipeline(models_service=self.ai_models)

    async def transcribe_audio(self, audio_file_path: str) -> StandardResponse:
        result = await self.ai_models.transcribe_audio(audio_file_path)
        return StandardResponse(
            success=result.get("success", False),
            data=(
                {
                    "transcription": result.get("transcription", ""),
                    "confidence": result.get("confidence", 0.0),
                }
                if result.get("success")
                else None
            ),
            error=result.get("error"),
            is_stub=result.get("is_stub", False),
            warning=result.get("warning"),
        )

    async def extract_entities(self, transcription: str) -> StandardResponse:
        result = await self.ai_models.extract_entities(transcription)
        return StandardResponse(
            success=result.get("success", False),
            data=(
                {
                    "entities": result.get("entities", {}),
                    "confidence": result.get("confidence", 0.0),
                }
                if result.get("success")
                else None
            ),
            error=result.get("error"),
            is_stub=result.get("is_stub", False),
            warning=result.get("warning"),
        )

    async def generate_note(
        self,
        transcription: str,
        entities: Dict[str, Any],
        *,
        consultation_id: Optional[str] = None,
        author_id: Optional[str] = None,
    ) -> StandardResponse:
        start = perf_counter()
        result = await self.ai_models.generate_note(transcription, entities)
        duration = perf_counter() - start

        if result.get("success") and consultation_id and result.get("generated_note"):
            version = self._persist_note(
                consultation_id=consultation_id,
                author_id=author_id,
                note_content=result.get("generated_note", ""),
                entities=entities,
                is_stub=result.get("is_stub", False),
            )
        else:
            version = None

        record_service_metric(
            service_name="scribe",
            metric_name="note_generation_seconds",
            metric_value=duration,
            metadata={
                "consultation_id": consultation_id,
                "success": result.get("success", False),
            },
        )
        record_llm_usage(
            request_id=None,
            user_id=author_id,
            model=result.get("model", self.ai_models.settings.gemini_model),
            tokens_prompt=result.get("tokens_prompt", 0),
            tokens_completion=result.get("tokens_completion", 0),
            cost_cents=result.get("cost_cents", 0.0),
            status="success" if result.get("success") else "failed",
        )

        return StandardResponse(
            success=result.get("success", False),
            data=(
                {
                    "note": result.get("generated_note", ""),
                    "confidence": result.get("confidence", 0.0),
                    "note_version_id": getattr(version, "id", None),
                }
                if result.get("success")
                else None
            ),
            error=result.get("error"),
            is_stub=result.get("is_stub", False),
            warning=result.get("warning"),
        )

    async def process_audio_pipeline(
        self,
        audio_file_path: str,
        *,
        consultation_id: Optional[str] = None,
        author_id: Optional[str] = None,
    ) -> StandardResponse:
        start = perf_counter()
        state = await self.pipeline.process_audio(audio_file_path)
        success = state.get("stage_completed") == "completed" and not state.get(
            "errors"
        )
        warnings = state.get("warnings", [])
        errors = state.get("errors", [])

        data = {
            "transcription": state.get("transcription", ""),
            "entities": state.get("entities", {}),
            "generated_note": state.get("generated_note", ""),
            "confidence_scores": state.get("confidence_scores", {}),
            "stage_completed": state.get("stage_completed"),
            "warnings": warnings,
        }

        if success and consultation_id and state.get("generated_note"):
            version = self._persist_note(
                consultation_id=consultation_id,
                author_id=author_id,
                note_content=state.get("generated_note", ""),
                entities=state.get("entities", {}),
                is_stub=state.get("is_stub", False),
            )
            data["note_version_id"] = getattr(version, "id", None)

        duration = perf_counter() - start
        record_service_metric(
            service_name="scribe",
            metric_name="pipeline_seconds",
            metric_value=duration,
            metadata={"consultation_id": consultation_id, "success": success},
        )

        return StandardResponse(
            success=success,
            data=data if success else None,
            error=errors[0] if errors else None,
            is_stub=state.get("is_stub", False),
            warning="; ".join(warnings) if warnings else None,
        )

    def _persist_note(
        self,
        *,
        consultation_id: str,
        author_id: Optional[str],
        note_content: str,
        entities: Dict[str, Any],
        is_stub: bool,
    ):
        with get_session() as session:
            version = crud.create_note_with_version(
                session,
                consultation_id=consultation_id,
                author_id=author_id,
                note_content=note_content,
                entities=entities,
                is_ai_generated=not is_stub,
            )
            session.refresh(version)
            return version
