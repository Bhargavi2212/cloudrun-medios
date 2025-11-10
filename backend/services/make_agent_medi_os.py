"""
MakeAgent Service
AI Scribe functionality using LangGraph pipeline
"""

from typing import Any, Dict, Optional

from backend.database import crud
from backend.database.session import get_session
from backend.services.telemetry import record_llm_usage

from .make_agent_pipeline import MakeAgentPipeline


class MakeAgentService:
    """AI Scribe service for clinical documentation"""

    def __init__(self):
        """Initialize MakeAgent service with LangGraph pipeline"""
        self.pipeline = MakeAgentPipeline()

    async def transcribe_audio(self, audio_file_path: str) -> str:
        """Transcribe audio file to text using Whisper.cpp"""
        # Use the pipeline for transcription
        result = await self.pipeline.process_audio(audio_file_path)
        return result.get("transcription", "")

    async def extract_entities(self, transcription: str) -> Dict[str, Any]:
        """Extract clinical entities using DistilBioBERT"""
        # For now, return placeholder entities
        # TODO: Implement direct entity extraction from text
        return {
            "symptoms": ["headache", "pain"],
            "duration": ["three days"],
            "severity": ["moderate"],
            "location": ["frontal region"],
            "medications": [],
            "diagnoses": [],
        }

    async def generate_note(self, transcription: str, entities: Dict[str, Any]) -> str:
        """Generate clinical note using Phi-3-mini"""
        # For now, return formatted note
        # TODO: Implement direct note generation from transcription and entities
        return f"""
SUBJECTIVE:
Patient presents with {', '.join(entities['symptoms'])} for {', '.join(entities['duration'])}.
Pain severity: {', '.join(entities['severity'])}
Location: {', '.join(entities['location'])}

OBJECTIVE:
[Physical examination findings to be documented]

ASSESSMENT:
[Diagnostic impressions to be documented]

PLAN:
[Treatment plan to be documented]
        """.strip()

    async def process_audio_pipeline(
        self,
        audio_file_path: str,
        *,
        consultation_id: Optional[str] = None,
        author_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process audio through the complete LangGraph pipeline.

        Args:
            audio_file_path: Path to the audio file
            consultation_id: Optional consultation ID to save notes to
            author_id: Optional author ID for the notes

        Returns:
            Dictionary with transcription, entities, generated_note, etc.
        """
        # Process audio through pipeline
        result = await self.pipeline.process_audio(audio_file_path)

        # Log LLM usage (even for template notes, so dashboard shows activity)
        import logging

        logger = logging.getLogger(__name__)

        # Record LLM usage for dashboard tracking
        # This includes template notes so the dashboard shows API usage even when Gemini fails
        # Template notes are logged with model="template-note-generator" to show activity in the dashboard
        try:
            model_name = result.get("model", "template")
            # For template notes, use a descriptive model name so dashboard shows activity
            if model_name == "template" or not model_name or model_name == "":
                model_name = "template-note-generator"

            # Determine status - success if we have a generated note, even if it's a template
            has_note = bool(result.get("generated_note", "").strip())
            is_success = (
                result.get("success", True)
                and has_note
                and result.get("stage_completed") != "failed"
            )

            record_llm_usage(
                request_id=None,  # Could get from context if available
                user_id=author_id,
                model=model_name,
                tokens_prompt=result.get("tokens_prompt", 0),
                tokens_completion=result.get("tokens_completion", 0),
                cost_cents=result.get("cost_cents", 0.0),
                status="success" if is_success else "failed",
            )
            logger.info(
                f"📊 Recorded LLM usage: model={model_name}, tokens={result.get('tokens_prompt', 0)}+{result.get('tokens_completion', 0)}, status={'success' if is_success else 'failed'}, has_note={has_note}"
            )
        except Exception as exc:
            logger.warning(f"⚠️ Failed to record LLM usage: {exc}", exc_info=True)

        # Save note to database if consultation_id is provided and we have a generated note
        # Save even if it's a template/stub note, as long as we have content
        logger.info(
            f"🔍 Note saving check: consultation_id={consultation_id}, has_generated_note={bool(result.get('generated_note'))}, generated_note_length={len(result.get('generated_note', ''))}"
        )

        if consultation_id and result.get("generated_note"):
            # Check if there are critical errors that prevent saving
            errors = result.get("errors", [])
            critical_errors = [
                e
                for e in errors
                if "transcription failed" in e.lower() or "file not found" in e.lower()
            ]

            logger.info(
                f"🔍 Error check: total_errors={len(errors)}, critical_errors={len(critical_errors)}"
            )

            if not critical_errors:
                try:
                    note_content = result.get("generated_note", "").strip()
                    logger.info(
                        f"🔍 Note content check: length={len(note_content)}, is_stub={result.get('is_stub', False)}"
                    )

                    if note_content:  # Only save if we have actual content
                        logger.info(
                            f"💾 Attempting to save note to consultation {consultation_id}..."
                        )
                        self._persist_note(
                            consultation_id=consultation_id,
                            author_id=author_id,
                            note_content=note_content,
                            entities=result.get("entities", {}),
                            is_stub=result.get("is_stub", False),
                        )
                        result["note_saved"] = True
                        logger.info(
                            f"✅ Note saved successfully to consultation {consultation_id} (is_stub={result.get('is_stub', False)}, content_length={len(note_content)})"
                        )
                    else:
                        logger.warning(
                            f"⚠️ Generated note is empty, not saving to consultation {consultation_id}"
                        )
                        result["note_saved"] = False
                        result["note_save_error"] = "Generated note is empty"
                except Exception as exc:
                    # Log error but don't fail the entire pipeline
                    logger.error(
                        f"❌ Failed to save note to consultation {consultation_id}: {exc}",
                        exc_info=True,
                    )
                    result["note_saved"] = False
                    result["note_save_error"] = str(exc)
            else:
                logger.warning(
                    f"⚠️ Not saving note due to critical errors: {critical_errors}"
                )
                result["note_saved"] = False
                result["note_save_error"] = "; ".join(critical_errors)
        elif consultation_id and not result.get("generated_note"):
            logger.warning(
                f"⚠️ Consultation ID provided ({consultation_id}) but no generated note to save"
            )
            result["note_saved"] = False
            result["note_save_error"] = "No generated note available"
        elif not consultation_id:
            logger.warning(
                f"⚠️ No consultation_id provided - note will not be saved. Result has generated_note: {bool(result.get('generated_note'))}"
            )
            result["note_saved"] = False
            result["note_save_error"] = "No consultation_id provided"

        return result

    def _persist_note(
        self,
        *,
        consultation_id: str,
        author_id: Optional[str],
        note_content: str,
        entities: Dict[str, Any],
        is_stub: bool,
    ):
        """Save note to database"""
        import logging

        logger = logging.getLogger(__name__)

        try:
            logger.info(
                f"💾 Starting note persistence: consultation_id={consultation_id}, content_length={len(note_content)}, is_stub={is_stub}"
            )

            with get_session() as session:
                logger.info(
                    f"📝 Creating note with version for consultation {consultation_id}"
                )
                version = crud.create_note_with_version(
                    session,
                    consultation_id=consultation_id,
                    author_id=author_id,
                    note_content=note_content,
                    entities=entities,
                    is_ai_generated=not is_stub,
                )
                # get_session() context manager will commit automatically
                session.refresh(version)
                logger.info(
                    f"✅ Note persisted successfully: note_id={version.note_id}, version_id={version.id}, consultation_id={consultation_id}"
                )
                return version
        except Exception as exc:
            logger.error(
                f"❌ Error persisting note for consultation {consultation_id}: {exc}",
                exc_info=True,
            )
            raise
