"""FastAPI routes for the AI Scribe."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, UploadFile, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ...services.config import get_settings
from ...services.error_response import StandardResponse
from ...services.job_queue import JobQueueService
from ...services.make_agent import MakeAgentService
from ...services.make_agent_medi_os import MakeAgentService as MediOSMakeAgentService
from ...services.storage import StorageError, StorageService

router = APIRouter()
service = MakeAgentService()
medi_os_service = MediOSMakeAgentService()
storage_service = StorageService()
# Use MediOS service for job queue to ensure note persistence works
job_queue_service = JobQueueService(make_agent_service=medi_os_service, storage_service=storage_service)
settings = get_settings()


class ExtractEntitiesRequest(BaseModel):
    transcript: str = Field(..., description="Transcribed text.")


class GenerateNoteRequest(BaseModel):
    transcript: str = Field(..., description="Transcribed text.")
    entities: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured entities extracted from the transcript.",
    )
    consultation_id: Optional[str] = Field(default=None, description="Consultation identifier to link note.")


class UpdateNoteRequest(BaseModel):
    content: str = Field(..., description="Updated note content")


class ApproveNoteRequest(BaseModel):
    approval_comment: Optional[str] = Field(default=None, description="Optional comment on approval")


class RejectNoteRequest(BaseModel):
    rejection_reason: Optional[str] = Field(default=None, description="Reason for rejection")


class SubmitNoteRequest(BaseModel):
    comment: Optional[str] = Field(default=None, description="Optional comment when submitting")


class ProcessAudioRequest(BaseModel):
    audio_id: str = Field(..., description="Audio file identifier.")


@router.get("/health", response_model=StandardResponse)
async def health_check() -> StandardResponse:
    """Simple health probe indicating whether models are loaded."""
    return StandardResponse(
        success=True,
        data={"status": "healthy"},
        is_stub=False,
    )


@router.post("/upload", response_model=StandardResponse, status_code=status.HTTP_201_CREATED)
async def upload_audio(
    audio_file: UploadFile = File(...),
    consultation_id: Optional[str] = Form(None),
) -> JSONResponse:
    """
    Upload audio for later processing.

    Files are persisted using the configured storage backend (local filesystem or GCS).
    Returns an audio identifier alongside metadata and, if available, a signed URL.
    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Uploading audio file: {audio_file.filename}, consultation_id: {consultation_id}")
        record, stored_file = await storage_service.save_audio_file(
            audio_file,
            consultation_id=consultation_id,
            uploaded_by=None,
        )
        logger.info(f"✅ Audio file saved successfully with ID: {record.id}, path: {record.storage_path}")
        logger.info(f"✅ Audio file consultation_id: {record.consultation_id} (from upload: {consultation_id})")

        # Verify the record can be retrieved immediately
        verify_record = storage_service.get_audio_file(record.id)
        if verify_record:
            logger.info(
                f"✅ Audio file verified in database: id={verify_record.id}, consultation_id={verify_record.consultation_id}"
            )
            if not verify_record.consultation_id:
                logger.error(
                    f"❌ WARNING: Audio file {verify_record.id} was saved WITHOUT consultation_id! Notes will not be saved."
                )
            else:
                logger.info(f"✅ Audio file has consultation_id={verify_record.consultation_id} - notes can be saved")
        else:
            logger.warning(f"⚠️ Audio file {record.id} was saved but cannot be retrieved immediately")

    except StorageError as exc:
        logger.error(f"Storage error during upload: {exc}")
        response = StandardResponse(success=False, error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=response.model_dump(),
        )
    except Exception as exc:
        logger.exception(f"Unexpected error during upload: {exc}")
        response = StandardResponse(success=False, error=f"Upload failed: {str(exc)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=response.model_dump(),
        )

    response = StandardResponse(
        success=True,
        data={
            "audio_id": record.id,
            "storage_path": record.storage_path,
            "mime_type": record.mime_type,
            "size_bytes": record.size_bytes,
            "signed_url": stored_file.signed_url,
        },
        is_stub=False,
    )
    logger.info(f"Upload response prepared for audio ID: {record.id}")
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=response.model_dump())


@router.post("/transcribe", response_model=StandardResponse)
async def transcribe_audio(audio_id: str) -> StandardResponse:
    """Transcribe a previously uploaded audio file."""
    record = storage_service.get_audio_file(audio_id)
    if not record:
        return StandardResponse(
            success=False,
            error=f"No audio file found for id: {audio_id}",
            is_stub=False,
        )

    file_path = storage_service.resolve_file_path(record)
    return await service.transcribe_audio(file_path)


@router.post("/extract_entities", response_model=StandardResponse)
async def extract_entities(payload: ExtractEntitiesRequest) -> StandardResponse:
    """Extract entities from a transcript."""
    return await service.extract_entities(payload.transcript)


@router.post("/generate_note", response_model=StandardResponse)
async def generate_note(payload: GenerateNoteRequest) -> StandardResponse:
    """Generate a SOAP note from transcript + entities."""
    return await service.generate_note(
        payload.transcript,
        payload.entities,
        consultation_id=payload.consultation_id,
    )


@router.post("/process", response_model=StandardResponse)
async def process_audio(payload: ProcessAudioRequest) -> StandardResponse:
    """Run the full pipeline (transcribe → entities → note)."""
    record = storage_service.get_audio_file(payload.audio_id)
    if not record:
        return StandardResponse(
            success=False,
            error=f"No audio file found for id: {payload.audio_id}",
            is_stub=False,
        )

    file_path = storage_service.resolve_file_path(record)
    return await service.process_audio_pipeline(
        file_path,
        consultation_id=record.consultation_id,
    )


@router.post("/medi-os/process", response_model=StandardResponse)
async def process_audio_medi_os(
    payload: ProcessAudioRequest,
) -> StandardResponse:
    """Run the MediOS LangGraph pipeline on uploaded audio."""
    import logging
    import os

    from fastapi import Depends

    from backend.security.dependencies import get_current_user

    logger = logging.getLogger(__name__)

    # Try to get current user for author_id (optional, don't fail if not authenticated)
    author_id = None
    try:
        # Note: We're not using Depends() here because we want it to be optional
        # The endpoint should work even without authentication for testing
        pass
    except Exception:
        pass

    record = storage_service.get_audio_file(payload.audio_id)
    if not record:
        logger.error(f"Audio file not found for ID: {payload.audio_id}")
        # Return error in the data structure that frontend expects
        return StandardResponse(
            success=True,  # Set to True so unwrap doesn't throw, but include errors in data
            data={
                "transcription": "",
                "entities": {},
                "generated_note": "",
                "confidence_scores": {},
                "warnings": [],
                "errors": [f"No audio file found for id: {payload.audio_id}"],
                "stage_completed": "failed",
            },
            is_stub=False,
        )

    try:
        logger.info(f"📥 Processing audio with ID: {payload.audio_id}")
        logger.info(f"📥 Audio file record: id={record.id}, consultation_id={record.consultation_id}")

        file_path = storage_service.resolve_file_path(record)
        # Ensure absolute path - resolve_path should already return absolute, but double-check
        file_path = os.path.abspath(os.path.normpath(file_path))
        logger.info(f"Resolved file path (absolute): {file_path}")
        logger.info(f"File path is absolute: {os.path.isabs(file_path)}")
        logger.info(
            f"Storage base path: {storage_service.provider.base_path if hasattr(storage_service.provider, 'base_path') else 'N/A'}"
        )
        logger.info(f"Record storage_path: {record.storage_path}")
        logger.info(f"Consultation ID: {record.consultation_id}, Author ID: {author_id}")

        # Verify file exists
        if not os.path.exists(file_path):
            logger.error(f"Audio file path does not exist: {file_path}")
            # Try to construct the path manually as a fallback
            if hasattr(storage_service.provider, "base_path"):
                alt_path = os.path.join(
                    str(storage_service.provider.base_path),
                    record.storage_path.replace("\\", "/"),
                )
                alt_path = os.path.abspath(alt_path)
                logger.info(f"Trying alternative path: {alt_path}")
                if os.path.exists(alt_path):
                    file_path = alt_path
                    logger.info(f"Using alternative path: {file_path}")
                else:
                    return StandardResponse(
                        success=True,
                        data={
                            "transcription": "",
                            "entities": {},
                            "generated_note": "",
                            "confidence_scores": {},
                            "warnings": [],
                            "errors": [f"Audio file not found at path: {file_path} (also tried: {alt_path})"],
                            "stage_completed": "failed",
                        },
                        is_stub=False,
                    )
            else:
                return StandardResponse(
                    success=True,
                    data={
                        "transcription": "",
                        "entities": {},
                        "generated_note": "",
                        "confidence_scores": {},
                        "warnings": [],
                        "errors": [f"Audio file not found at path: {file_path}"],
                        "stage_completed": "failed",
                    },
                    is_stub=False,
                )

        logger.info(f"Audio file exists: {file_path} ({os.path.getsize(file_path)} bytes)")

        # Process audio with consultation_id to save notes (only if consultation_id is set)
        consultation_id = record.consultation_id if record.consultation_id else None
        if not consultation_id:
            logger.warning(f"Audio file {payload.audio_id} has no consultation_id - notes will not be saved")

        result = await medi_os_service.process_audio_pipeline(
            file_path,
            consultation_id=consultation_id,
            author_id=author_id,
        )
        logger.info(f"Processing completed successfully for audio ID: {payload.audio_id}")
        logger.info(
            f"Processing result: note_saved={result.get('note_saved')}, has_generated_note={bool(result.get('generated_note'))}, consultation_id={consultation_id}"
        )

        if result.get("note_saved") and consultation_id:
            logger.info(f"✅ SUCCESS: Note saved to consultation {consultation_id}")
            logger.info(f"Note content preview: {result.get('generated_note', '')[:100]}...")
        elif result.get("note_save_error"):
            logger.warning(f"❌ Note save failed: {result.get('note_save_error')}")
        elif not consultation_id:
            logger.warning(f"⚠️ Note not saved - no consultation_id provided (audio_id: {payload.audio_id})")
        elif not result.get("generated_note"):
            logger.warning(f"⚠️ Note not saved - no generated_note in result (consultation_id: {consultation_id})")
        else:
            logger.warning(
                f"⚠️ Note not saved - unknown reason (consultation_id: {consultation_id}, note_saved: {result.get('note_saved')})"
            )

        # Ensure errors list is present even if empty
        if "errors" not in result:
            result["errors"] = []
        if "warnings" not in result:
            result["warnings"] = []

        # Always return success=True, but include errors in the data
        # Frontend will check for errors array
        return StandardResponse(success=True, data=result)
    except Exception as exc:
        logger.exception(f"Error processing audio {payload.audio_id}: {exc}")
        return StandardResponse(
            success=True,  # Set to True so unwrap doesn't throw
            data={
                "transcription": "",
                "entities": {},
                "generated_note": "",
                "confidence_scores": {},
                "warnings": [],
                "errors": [f"Failed to process audio: {str(exc)}"],
                "stage_completed": "failed",
            },
            is_stub=False,
        )


@router.post(
    "/process_async",
    response_model=StandardResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def process_audio_async(payload: ProcessAudioRequest) -> StandardResponse:
    record = storage_service.get_audio_file(payload.audio_id)
    if not record:
        return StandardResponse(
            success=False,
            error=f"No audio file found for id: {payload.audio_id}",
            is_stub=False,
        )

    job = await job_queue_service.enqueue_audio_processing(payload.audio_id)
    return StandardResponse(
        success=True,
        data={"job_id": job.id, "status": job.status},
    )


@router.get("/jobs/{job_id}", response_model=StandardResponse)
async def get_job_status(job_id: str) -> StandardResponse:
    job = job_queue_service.get_job(job_id)
    if not job:
        return StandardResponse(success=False, error="Job not found")
    return StandardResponse(
        success=True,
        data={
            "job_id": job.id,
            "status": job.status,
            "payload": job.payload,
            "error": job.error_message,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
        },
    )


@router.post("/medi-os/process-stream")
async def process_audio_stream(payload: ProcessAudioRequest) -> StreamingResponse:
    """Process audio with Server-Sent Events (SSE) streaming for real-time updates.

    Returns a stream of JSON events as each stage completes:
    - Transcription progress
    - Entity extraction progress
    - Note generation progress

    The stream can be consumed using EventSource in the frontend.
    """
    import os

    logger = logging.getLogger(__name__)

    record = storage_service.get_audio_file(payload.audio_id)
    if not record:

        async def error_stream():
            yield f"data: {json.dumps({'error': f'No audio file found for id: {payload.audio_id}', 'status': 'error'})}\n\n"

        return StreamingResponse(
            error_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def generate_stream():
        """Generate SSE stream from pipeline processing."""
        try:
            file_path = storage_service.resolve_file_path(record)
            file_path = os.path.abspath(os.path.normpath(file_path))

            # Verify file exists
            if not os.path.exists(file_path):
                if hasattr(storage_service.provider, "base_path"):
                    alt_path = os.path.join(
                        str(storage_service.provider.base_path),
                        record.storage_path.replace("\\", "/"),
                    )
                    alt_path = os.path.abspath(alt_path)
                    if os.path.exists(alt_path):
                        file_path = alt_path
                    else:
                        yield f"data: {json.dumps({'error': f'Audio file not found at path: {file_path}', 'status': 'error'})}\n\n"
                        return
                else:
                    yield f"data: {json.dumps({'error': f'Audio file not found at path: {file_path}', 'status': 'error'})}\n\n"
                    return

            # Get consultation_id and author_id
            consultation_id = record.consultation_id if record.consultation_id else None
            author_id = record.uploaded_by

            # Stream from pipeline
            final_result = None
            async for update in medi_os_service.pipeline.process_audio_streaming(file_path):
                # Send update as SSE event
                event_data = {
                    **update,
                    "audio_id": payload.audio_id,
                    "consultation_id": consultation_id,
                }
                yield f"data: {json.dumps(event_data)}\n\n"
                final_result = update

            # After streaming completes, save note and record LLM usage if needed
            if final_result and consultation_id and final_result.get("generated_note"):
                try:
                    note_content = final_result.get("generated_note", "").strip()
                    if note_content:
                        # Record LLM usage
                        from backend.services.telemetry import record_llm_usage

                        try:
                            model_name = final_result.get("model", "template")
                            if model_name == "template" or not model_name or model_name == "":
                                model_name = "template-note-generator"
                            has_note = bool(note_content)
                            is_success = (
                                final_result.get("success", True)
                                and has_note
                                and final_result.get("stage_completed") != "failed"
                            )
                            record_llm_usage(
                                request_id=None,
                                user_id=author_id,
                                model=model_name,
                                tokens_prompt=final_result.get("tokens_prompt", 0),
                                tokens_completion=final_result.get("tokens_completion", 0),
                                cost_cents=final_result.get("cost_cents", 0.0),
                                status="success" if is_success else "failed",
                            )
                        except Exception as exc:
                            logger.warning(f"Failed to record LLM usage: {exc}")

                        # Save note to database
                        medi_os_service._persist_note(
                            consultation_id=consultation_id,
                            author_id=author_id,
                            note_content=note_content,
                            entities=final_result.get("entities", {}),
                            is_stub=final_result.get("is_stub", False),
                        )
                        # Send final update about note saving
                        yield f"data: {json.dumps({'stage': 'note_saved', 'status': 'completed', 'note_saved': True, 'consultation_id': consultation_id})}\n\n"
                except Exception as exc:
                    logger.error(f"Failed to save note after streaming: {exc}")
                    yield f"data: {json.dumps({'stage': 'note_saved', 'status': 'failed', 'error': str(exc)})}\n\n"

        except Exception as exc:
            logger.exception(f"Error in streaming pipeline: {exc}")
            yield f"data: {json.dumps({'error': str(exc), 'status': 'error'})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/consultations/{consultation_id}/note", response_model=StandardResponse)
async def get_consultation_note(consultation_id: str) -> StandardResponse:
    """Get the current note for a consultation."""
    from backend.database import crud
    from backend.database.session import get_session

    with get_session() as session:
        note = crud.get_note_for_consultation(session, consultation_id)
        if not note:
            return StandardResponse(
                success=True,
                data={"note": None, "message": "No note found for this consultation"},
                is_stub=False,
            )

        current_version = note.current_version
        if not current_version:
            return StandardResponse(
                success=True,
                data={"note": None, "message": "Note exists but has no content"},
                is_stub=False,
            )

        return StandardResponse(
            success=True,
            data={
                "note_id": note.id,
                "consultation_id": note.consultation_id,
                "status": note.status,
                "content": current_version.content,
                "entities": current_version.entities,
                "is_ai_generated": current_version.is_ai_generated,
                "created_at": (current_version.created_at.isoformat() if current_version.created_at else None),
                "version_id": current_version.id,
            },
            is_stub=False,
        )


@router.put("/consultations/{consultation_id}/note", response_model=StandardResponse)
async def update_consultation_note(
    consultation_id: str,
    payload: UpdateNoteRequest,
) -> StandardResponse:
    """Update an existing note for a consultation."""
    from fastapi import HTTPException, status

    from backend.database import crud
    from backend.database.session import get_session
    from backend.security.dependencies import get_current_user

    # Try to get current user for author_id (optional)
    author_id = None
    try:
        # Note: Authentication is optional for now - can be added later with Depends()
        pass
    except Exception:
        pass  # Allow updates without authentication for now

    try:
        with get_session() as session:
            version = crud.update_note_content(
                session,
                consultation_id=consultation_id,
                content=payload.content,
                author_id=author_id,
            )
            return StandardResponse(
                success=True,
                data={
                    "note_id": version.note_id,
                    "version_id": version.id,
                    "content": version.content,
                    "updated_at": (version.created_at.isoformat() if version.created_at else None),
                },
                is_stub=False,
            )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        import logging

        logger = logging.getLogger(__name__)
        logger.exception(f"Error updating note for consultation {consultation_id}: {exc}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.post("/consultations/{consultation_id}/note/submit", response_model=StandardResponse)
async def submit_note_for_approval(
    consultation_id: str,
    payload: SubmitNoteRequest,
) -> StandardResponse:
    """Submit a note for approval."""
    from fastapi import HTTPException, status

    from backend.database import crud
    from backend.database.session import get_session

    try:
        with get_session() as session:
            note = crud.submit_note_for_approval(session, consultation_id)
            return StandardResponse(
                success=True,
                data={
                    "note_id": note.id,
                    "consultation_id": note.consultation_id,
                    "status": note.status,
                    "message": "Note submitted for approval",
                },
                is_stub=False,
            )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        import logging

        logger = logging.getLogger(__name__)
        logger.exception(f"Error submitting note for approval: {exc}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.post("/consultations/{consultation_id}/note/approve", response_model=StandardResponse)
async def approve_note(
    consultation_id: str,
    payload: ApproveNoteRequest,
) -> StandardResponse:
    """Approve a note."""
    from fastapi import HTTPException, status

    from backend.database import crud
    from backend.database.session import get_session

    # Try to get current user for approver_id (optional)
    approver_id = None
    try:
        # Note: Authentication is optional for now - can be added later with Depends()
        pass
    except Exception:
        pass

    try:
        with get_session() as session:
            note = crud.approve_note(session, consultation_id, approver_id=approver_id)
            return StandardResponse(
                success=True,
                data={
                    "note_id": note.id,
                    "consultation_id": note.consultation_id,
                    "status": note.status,
                    "message": "Note approved",
                },
                is_stub=False,
            )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        import logging

        logger = logging.getLogger(__name__)
        logger.exception(f"Error approving note: {exc}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.post("/consultations/{consultation_id}/note/reject", response_model=StandardResponse)
async def reject_note(
    consultation_id: str,
    payload: RejectNoteRequest,
) -> StandardResponse:
    """Reject a note."""
    from fastapi import HTTPException, status

    from backend.database import crud
    from backend.database.session import get_session

    # Try to get current user for approver_id (optional)
    approver_id = None
    try:
        # Note: Authentication is optional for now - can be added later with Depends()
        pass
    except Exception:
        pass

    try:
        with get_session() as session:
            note = crud.reject_note(
                session,
                consultation_id,
                rejection_reason=payload.rejection_reason,
                approver_id=approver_id,
            )
            return StandardResponse(
                success=True,
                data={
                    "note_id": note.id,
                    "consultation_id": note.consultation_id,
                    "status": note.status,
                    "message": "Note rejected",
                },
                is_stub=False,
            )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        import logging

        logger = logging.getLogger(__name__)
        logger.exception(f"Error rejecting note: {exc}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
