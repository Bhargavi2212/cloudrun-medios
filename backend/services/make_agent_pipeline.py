"""
LangGraph-inspired pipeline for the AI Scribe.

Stages:
  1. Transcription via Whisper.
  2. Entity extraction via lightweight rules.
  3. SOAP note generation via Gemini (with graceful fallback).
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

from pydantic import BaseModel, Field

from .ai_models import AIModelsService


class MakeAgentState(BaseModel):
    audio_file_path: str
    transcription: str = ""
    entities: Dict[str, Any] = Field(default_factory=dict)
    generated_note: str = ""
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    stage_completed: str = ""
    is_stub: bool = False


class MakeAgentPipeline:
    """Three-stage AI pipeline for clinical documentation."""

    def __init__(self, models_service: Optional[AIModelsService] = None) -> None:
        self.ai_models = models_service or AIModelsService()

    async def process_audio_streaming(
        self, audio_file_path: str
    ) -> AsyncIterator[Dict[str, Any]]:
        """Process audio with streaming updates for each stage.
        
        Yields updates as each stage completes:
        - {"stage": "transcription", "status": "in_progress", ...}
        - {"stage": "transcription", "status": "completed", "transcription": "...", ...}
        - {"stage": "entity_extraction", "status": "in_progress", ...}
        - {"stage": "entity_extraction", "status": "completed", "entities": {...}, ...}
        - {"stage": "note_generation", "status": "in_progress", ...}
        - {"stage": "note_generation", "status": "completed", "generated_note": "...", ...}
        """
        state = MakeAgentState(audio_file_path=audio_file_path)
        
        # Stage 1: Transcription
        yield {
            "stage": "transcription",
            "status": "in_progress",
            "progress": 0.33,
            "message": "Transcribing audio...",
        }
        
        transcription_result = await self.ai_models.transcribe_audio(audio_file_path)
        state.confidence_scores["transcription"] = transcription_result.get("confidence", 0.0)
        state.warnings.extend(_collect_optional(transcription_result.get("warning")))
        state.is_stub = state.is_stub or transcription_result.get("is_stub", False)
        
        if not transcription_result.get("success"):
            state.errors.append(transcription_result.get("error", "Transcription failed."))
            state.stage_completed = "transcription_failed"
            yield {
                "stage": "transcription",
                "status": "failed",
                "progress": 0.33,
                "error": transcription_result.get("error", "Transcription failed."),
                **state.model_dump(),
            }
            return
        
        state.transcription = transcription_result.get("transcription", "")
        yield {
            "stage": "transcription",
            "status": "completed",
            "progress": 0.33,
            "transcription": state.transcription,
            "confidence": state.confidence_scores.get("transcription", 0.0),
            **state.model_dump(),
        }
        
        # Stage 2: Entity extraction
        yield {
            "stage": "entity_extraction",
            "status": "in_progress",
            "progress": 0.66,
            "message": "Extracting entities...",
        }
        
        entity_result = await self.ai_models.extract_entities(state.transcription)
        state.confidence_scores["entity_extraction"] = entity_result.get("confidence", 0.0)
        state.warnings.extend(_collect_optional(entity_result.get("warning")))
        state.is_stub = state.is_stub or entity_result.get("is_stub", False)
        
        if not entity_result.get("success"):
            state.errors.append(entity_result.get("error", "Entity extraction failed."))
            state.stage_completed = "entity_extraction_failed"
            yield {
                "stage": "entity_extraction",
                "status": "failed",
                "progress": 0.66,
                "error": entity_result.get("error", "Entity extraction failed."),
                **state.model_dump(),
            }
            return
        
        state.entities = entity_result.get("entities", {})
        yield {
            "stage": "entity_extraction",
            "status": "completed",
            "progress": 0.66,
            "entities": state.entities,
            "confidence": state.confidence_scores.get("entity_extraction", 0.0),
            **state.model_dump(),
        }
        
        # Stage 3: Note generation
        yield {
            "stage": "note_generation",
            "status": "in_progress",
            "progress": 0.90,
            "message": "Generating clinical note...",
        }
        
        note_result = await self.ai_models.generate_note(state.transcription, state.entities)
        state.confidence_scores["note_generation"] = note_result.get("confidence", 0.0)
        state.warnings.extend(_collect_optional(note_result.get("warning")))
        state.is_stub = state.is_stub or note_result.get("is_stub", False)
        
        if not note_result.get("success"):
            state.errors.append(note_result.get("error", "Note generation failed."))
            state.stage_completed = "note_generation_failed"
            result_dict = state.model_dump()
            result_dict["model"] = note_result.get("model", "template")
            result_dict["tokens_prompt"] = note_result.get("tokens_prompt", 0)
            result_dict["tokens_completion"] = note_result.get("tokens_completion", 0)
            result_dict["cost_cents"] = note_result.get("cost_cents", 0.0)
            result_dict["success"] = False
            yield {
                "stage": "note_generation",
                "status": "failed",
                "progress": 1.0,
                "error": note_result.get("error", "Note generation failed."),
                **result_dict,
            }
            return
        
        state.generated_note = note_result.get("generated_note", "")
        state.stage_completed = "completed"
        result_dict = state.model_dump()
        result_dict["model"] = note_result.get("model", "template")
        result_dict["tokens_prompt"] = note_result.get("tokens_prompt", 0)
        result_dict["tokens_completion"] = note_result.get("tokens_completion", 0)
        result_dict["cost_cents"] = note_result.get("cost_cents", 0.0)
        result_dict["success"] = note_result.get("success", True)
        
        yield {
            "stage": "note_generation",
            "status": "completed",
            "progress": 1.0,
            "generated_note": state.generated_note,
            "confidence": state.confidence_scores.get("note_generation", 0.0),
            **result_dict,
        }

    async def process_audio(self, audio_file_path: str) -> Dict[str, Any]:
        state = MakeAgentState(audio_file_path=audio_file_path)

        # Stage 1: Transcription
        transcription_result = await self.ai_models.transcribe_audio(audio_file_path)
        state.confidence_scores["transcription"] = transcription_result.get("confidence", 0.0)
        state.warnings.extend(_collect_optional(transcription_result.get("warning")))
        state.is_stub = state.is_stub or transcription_result.get("is_stub", False)
        if not transcription_result.get("success"):
            state.errors.append(transcription_result.get("error", "Transcription failed."))
            state.stage_completed = "transcription_failed"
            return state.model_dump()

        state.transcription = transcription_result.get("transcription", "")

        # Stage 2: Entity extraction
        entity_result = await self.ai_models.extract_entities(state.transcription)
        state.confidence_scores["entity_extraction"] = entity_result.get("confidence", 0.0)
        state.warnings.extend(_collect_optional(entity_result.get("warning")))
        state.is_stub = state.is_stub or entity_result.get("is_stub", False)
        if not entity_result.get("success"):
            state.errors.append(entity_result.get("error", "Entity extraction failed."))
            state.stage_completed = "entity_extraction_failed"
            return state.model_dump()

        state.entities = entity_result.get("entities", {})

        # Stage 3: Note generation
        note_result = await self.ai_models.generate_note(state.transcription, state.entities)
        state.confidence_scores["note_generation"] = note_result.get("confidence", 0.0)
        state.warnings.extend(_collect_optional(note_result.get("warning")))
        state.is_stub = state.is_stub or note_result.get("is_stub", False)
        if not note_result.get("success"):
            state.errors.append(note_result.get("error", "Note generation failed."))
            state.stage_completed = "note_generation_failed"
            # Include model/token info even on failure for tracking
            result_dict = state.model_dump()
            result_dict["model"] = note_result.get("model", "template")
            result_dict["tokens_prompt"] = note_result.get("tokens_prompt", 0)
            result_dict["tokens_completion"] = note_result.get("tokens_completion", 0)
            result_dict["cost_cents"] = note_result.get("cost_cents", 0.0)
            result_dict["success"] = False
            return result_dict

        state.generated_note = note_result.get("generated_note", "")
        state.stage_completed = "completed"
        # Include model/token info in the result for LLM usage tracking
        result_dict = state.model_dump()
        result_dict["model"] = note_result.get("model", "template")
        result_dict["tokens_prompt"] = note_result.get("tokens_prompt", 0)
        result_dict["tokens_completion"] = note_result.get("tokens_completion", 0)
        result_dict["cost_cents"] = note_result.get("cost_cents", 0.0)
        result_dict["success"] = note_result.get("success", True)
        return result_dict


def _collect_optional(value: Optional[str]) -> List[str]:
    if value:
        return [value]
    return []

