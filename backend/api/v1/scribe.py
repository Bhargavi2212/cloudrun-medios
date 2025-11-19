from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.database import crud
from backend.database.models import ScribeSessionStatus, ScribeTranscriptSegment, SoapNote
from backend.database.schemas import (
    ScribeSegmentRead,
    ScribeSessionRead,
    ScribeVitalRead,
    SoapNoteRead,
)
from backend.database.session import get_session
from backend.security.dependencies import require_roles
from backend.security.permissions import UserRole
from backend.services.ai_scribe import audio_gateway_service, soap_summarizer, triage_bridge
from backend.services.ai_scribe.exporter import build_fhir_document, build_pdf
from backend.services.error_response import StandardResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scribe", tags=["scribe"])


class ScribeSessionCreateRequest(BaseModel):
    consultation_id: UUID = Field(..., description="Consultation associated with the encounter.")
    patient_id: UUID = Field(..., description="Patient identifier.")
    language: Optional[str] = Field(default="en")
    metadata: Optional[dict] = Field(default=None)


class VitalsCreateRequest(BaseModel):
    recorded_at: Optional[datetime] = None
    heart_rate: Optional[int] = Field(default=None, ge=0, le=240)
    respiratory_rate: Optional[int] = Field(default=None, ge=0, le=80)
    systolic_bp: Optional[int] = Field(default=None, ge=0, le=300)
    diastolic_bp: Optional[int] = Field(default=None, ge=0, le=200)
    temperature_c: Optional[float] = Field(default=None, ge=30.0, le=43.0)
    oxygen_saturation: Optional[int] = Field(default=None, ge=0, le=100)
    pain_score: Optional[int] = Field(default=None, ge=0, le=10)
    source: str = Field(default="manual")


class SegmentUpdateRequest(BaseModel):
    speaker_label: Optional[str] = None
    text: Optional[str] = None


class SoapNoteUpdateRequest(BaseModel):
    content: Optional[dict] = None
    status: Optional[str] = None


@router.post("/sessions", response_model=StandardResponse, status_code=201)
async def create_scribe_session(
    payload: ScribeSessionCreateRequest,
    user=Depends(require_roles(UserRole.NURSE, UserRole.DOCTOR, UserRole.ADMIN)),
):
    with get_session() as db:
        session = crud.create_scribe_session(
            db,
            consultation_id=str(payload.consultation_id),
            patient_id=str(payload.patient_id),
            created_by=str(getattr(user, "id", "")) if user else None,
            status=ScribeSessionStatus.CREATED,
            language=payload.language,
            session_metadata=payload.metadata,
        )
        data = {"session": ScribeSessionRead.model_validate(session).model_dump()}
    return StandardResponse(success=True, data=data)


@router.get(
    "/sessions/{session_id}",
    response_model=StandardResponse,
    dependencies=[Depends(require_roles(UserRole.NURSE, UserRole.DOCTOR, UserRole.ADMIN))],
)
async def get_session_details(session_id: str):
    with get_session() as db:
        session = crud.get_scribe_session(db, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Scribe session not found")
        segments = crud.list_transcript_segments(db, session_id, limit=500)
        vitals = crud.list_scribe_vitals(db, session_id, limit=200)
        notes = crud.list_soap_notes_for_session(db, session_id)
        triage_predictions = crud.list_triage_predictions_for_session(db, session_id)
    return StandardResponse(
        success=True,
        data={
            "session": ScribeSessionRead.model_validate(session).model_dump(),
            "segments": [ScribeSegmentRead.model_validate(seg).model_dump() for seg in segments],
            "vitals": [ScribeVitalRead.model_validate(entry).model_dump() for entry in vitals],
            "notes": [SoapNoteRead.model_validate(note).model_dump() for note in notes],
            "triage_predictions": [
                {
                    "id": prediction.id,
                    "esi_level": prediction.esi_level,
                    "probability": float(prediction.probability or 0.0),
                    "probabilities": prediction.probabilities,
                    "flagged": prediction.flagged,
                    "created_at": prediction.created_at.isoformat(),
                }
                for prediction in triage_predictions
            ],
        },
    )


@router.post("/sessions/{session_id}/vitals", response_model=StandardResponse)
async def record_vitals(
    session_id: str,
    payload: VitalsCreateRequest,
    user=Depends(require_roles(UserRole.NURSE, UserRole.DOCTOR)),
):
    recorded_at = payload.recorded_at or datetime.now(timezone.utc)
    with get_session() as db:
        session = crud.get_scribe_session(db, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Scribe session not found")
        vitals = crud.record_scribe_vitals(
            db,
            session_id=session_id,
            recorded_at=recorded_at,
            source=payload.source,
            recorded_by=str(getattr(user, "id", "")) if user else None,
            heart_rate=payload.heart_rate,
            respiratory_rate=payload.respiratory_rate,
            systolic_bp=payload.systolic_bp,
            diastolic_bp=payload.diastolic_bp,
            temperature_c=payload.temperature_c,
            oxygen_saturation=payload.oxygen_saturation,
            pain_score=payload.pain_score,
        )
    await audio_gateway_service.publish_event(
        session_id,
        "scribe.vitals.recorded",
        ScribeVitalRead.model_validate(vitals).model_dump(),
    )
    return StandardResponse(success=True, data={"vitals": ScribeVitalRead.model_validate(vitals).model_dump()})


@router.post(
    "/sessions/{session_id}/complete",
    response_model=StandardResponse,
    dependencies=[Depends(require_roles(UserRole.NURSE, UserRole.DOCTOR, UserRole.ADMIN))],
)
async def finalize_session(session_id: str):
    with get_session() as db:
        session = crud.get_scribe_session(db, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Scribe session not found")
        transcript = crud.list_transcript_segments(db, session_id, limit=5)
        if not transcript:
            raise HTTPException(status_code=400, detail="Cannot finalize session without transcript data.")
    note_result = await soap_summarizer.summarize_session(session_id)
    triage_result = triage_bridge.predict_for_session(session_id)
    await audio_gateway_service.publish_event(
        session_id,
        "scribe.note.generated",
        {"note": note_result, "triage": triage_result},
    )
    return StandardResponse(success=True, data={"soap_note": note_result, "triage": triage_result})


@router.post(
    "/sessions/{session_id}/triage",
    response_model=StandardResponse,
    dependencies=[Depends(require_roles(UserRole.NURSE, UserRole.DOCTOR))],
)
async def run_triage(session_id: str):
    triage_result = triage_bridge.predict_for_session(session_id)
    if triage_result is None:
        raise HTTPException(status_code=503, detail="Triage model unavailable")
    await audio_gateway_service.publish_event(session_id, "scribe.triage.updated", triage_result)
    return StandardResponse(success=True, data=triage_result)


@router.patch(
    "/sessions/{session_id}/segments/{segment_id}",
    response_model=StandardResponse,
)
async def update_segment(
    session_id: str,
    segment_id: str,
    payload: SegmentUpdateRequest,
    _user=Depends(require_roles(UserRole.NURSE, UserRole.DOCTOR, UserRole.ADMIN)),
):
    with get_session() as db:
        segment = db.get(ScribeTranscriptSegment, segment_id)
        if segment is None:
            raise HTTPException(status_code=404, detail="Segment not found")
        if segment.session_id != session_id:
            raise HTTPException(status_code=400, detail="Segment does not belong to session")
        if payload.speaker_label is not None:
            segment.speaker_label = payload.speaker_label
        if payload.text is not None:
            segment.text = payload.text
        db.add(segment)
    await audio_gateway_service.publish_event(
        session_id,
        "scribe.transcript.corrected",
        {"id": segment_id, "speaker": segment.speaker_label, "text": segment.text},
    )
    return StandardResponse(success=True, data={"segment": ScribeSegmentRead.model_validate(segment).model_dump()})


@router.websocket("/sessions/{session_id}/audio")
async def audio_stream(session_id: str, websocket: WebSocket):
    await audio_gateway_service.handle_audio_stream(session_id, websocket)


@router.websocket("/sessions/{session_id}/events")
async def event_stream(session_id: str, websocket: WebSocket):
    await audio_gateway_service.stream_events(session_id, websocket)


@router.get(
    "/sessions/{session_id}/export/pdf",
    response_class=StreamingResponse,
    dependencies=[Depends(require_roles(UserRole.NURSE, UserRole.DOCTOR, UserRole.ADMIN))],
)
async def export_pdf(session_id: str):
    with get_session() as db:
        notes = crud.list_soap_notes_for_session(db, session_id)
        if not notes:
            raise HTTPException(status_code=404, detail="No SOAP note found for session")
        note_schema = SoapNoteRead.model_validate(notes[0])
        session = crud.get_scribe_session(db, session_id)
        session_info = {"consultation_id": session.consultation_id if session else None, "patient_id": session.patient_id if session else None}
    pdf_bytes = build_pdf(note_schema, session_info=session_info)
    return StreamingResponse(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="soap-note-{session_id}.pdf"'},
    )


@router.get(
    "/sessions/{session_id}/export/fhir",
    response_class=JSONResponse,
    dependencies=[Depends(require_roles(UserRole.NURSE, UserRole.DOCTOR, UserRole.ADMIN))],
)
async def export_fhir(session_id: str):
    with get_session() as db:
        notes = crud.list_soap_notes_for_session(db, session_id)
        if not notes:
            raise HTTPException(status_code=404, detail="No SOAP note found for session")
        note_schema = SoapNoteRead.model_validate(notes[0])
        session = crud.get_scribe_session(db, session_id)
        session_info = {"consultation_id": session.consultation_id if session else None, "patient_id": session.patient_id if session else None}
    document = build_fhir_document(note_schema, session_info=session_info)
    return JSONResponse(content=document)


@router.put(
    "/notes/{note_id}",
    response_model=StandardResponse,
)
async def update_soap_note(
    note_id: str,
    payload: SoapNoteUpdateRequest,
    _user=Depends(require_roles(UserRole.NURSE, UserRole.DOCTOR, UserRole.ADMIN)),
):
    if payload.content is None and payload.status is None:
        raise HTTPException(status_code=400, detail="No updates supplied.")
    with get_session() as db:
        note = db.get(SoapNote, note_id)
        if note is None:
            raise HTTPException(status_code=404, detail="SOAP note not found.")
        updates = {}
        if payload.content is not None:
            updates["content"] = payload.content
            updates["raw_markdown"] = _content_to_markdown(payload.content)
        if payload.status is not None:
            updates["status"] = payload.status
        updated = crud.update_soap_note(db, note, **updates)
    return StandardResponse(success=True, data={"note": SoapNoteRead.model_validate(updated).model_dump()})


def _content_to_markdown(content: dict) -> str:
    def render_section(title: str, data: Optional[dict]) -> str:
        if not data:
            return f"## {title}\n_Not documented_\n"
        summary = data.get("summary") or data.get("primary_impression") or ""
        lines = [summary] if summary else []
        for key in ("details", "exam", "diagnostics", "therapies"):
            for entry in data.get(key, []):
                lines.append(f"- {entry}")
        return f"## {title}\n" + ("\n".join(filter(None, lines)) or "_Not documented_") + "\n"

    sections = [
        render_section("Subjective", content.get("subjective")),
        render_section("Objective", content.get("objective")),
        render_section("Assessment", content.get("assessment")),
        render_section("Plan", content.get("plan")),
    ]
    return "\n".join(sections)

