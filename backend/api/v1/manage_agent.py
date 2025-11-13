from __future__ import annotations

import asyncio
import json
import os
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.database import crud
from backend.database.models import (
    Consultation,
    ConsultationStatus,
    DocumentProcessingStatus,
    FileAsset,
    Patient,
    QueueStage,
    QueueState,
    TimelineEventStatus,
    User,
    Vital,
)
from backend.database.session import get_db_session, get_session
from backend.dto.manage_agent_dto import (
    CheckInRequest,
    ConsultationRecord,
    DocumentReviewRequest,
    DocumentReviewResolution,
    DocumentStatusResponse,
    PatientQueueItem,
    QueueResponse,
    TimelineSummaryResponse,
    TriageResult,
    VitalsSubmission,
)
from backend.security.dependencies import require_roles
from backend.security.permissions import UserRole
from backend.services.document_processing import DocumentProcessingService
from backend.services.error_response import StandardResponse
from backend.services.manage_agent_state_machine import ManageAgentStateMachine
from backend.services.manage_agent_wait_time import ManageAgentWaitTimePredictor
from backend.services.notifier import notification_service
from backend.services.queue_service import queue_service
from backend.services.storage import StorageService
from backend.services.timeline_summary_service import TimelineSummaryService
from backend.services.triage_engine import TriageEngine

router = APIRouter(prefix="/manage", tags=["manage-agent"])

triage_engine = TriageEngine()
state_machine = ManageAgentStateMachine()
wait_time_predictor = ManageAgentWaitTimePredictor()
storage_service = StorageService()
timeline_summary_service = TimelineSummaryService()
document_processing_service = DocumentProcessingService(storage_service=storage_service)


@router.post(
    "/check-in",
    response_model=StandardResponse,
    status_code=status.HTTP_201_CREATED,
)
def check_in_patient(
    request: CheckInRequest,
    session: Session = Depends(get_db_session),
    _: User = Depends(require_roles(UserRole.RECEPTIONIST, UserRole.NURSE, UserRole.ADMIN)),
) -> StandardResponse:
    """Check a patient into the queue and compute an initial triage estimate."""
    patient = (
        session.query(Patient)
        .filter(
            Patient.id == request.patient_id,
            Patient.is_deleted.is_(False),
        )
        .first()
    )
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Run a lightweight triage estimate to determine initial priority
    preliminary_state = state_machine.process_patient_check_in(
        consultation_id="pending",
        patient_id=patient.id,
        chief_complaint=request.chief_complaint,
    )
    initial_priority = preliminary_state.triage_level or 3

    queue_state = queue_service.create_entry(
        patient_id=patient.id,
        consultation_id=None,
        chief_complaint=request.chief_complaint,
        priority_level=initial_priority,
        estimated_wait_seconds=None,
        assigned_to=None,
        created_by=None,
    )

    consultation_id = queue_state.consultation_id or (
        queue_state.consultation.id if queue_state.consultation else queue_state.id
    )
    preliminary_state.consultation_id = consultation_id

    if queue_state.consultation_id:
        db_consultation = crud.get_consultation(session, queue_state.consultation_id)
        if db_consultation:
            crud.update_consultation(
                session,
                db_consultation,
                triage_level=initial_priority,
            )
            session.commit()

    # Refresh session view of queue data for downstream calculations
    session.expire_all()
    queue_items = _build_queue_items(session)
    queue_snapshot = next(
        (item for item in queue_items if item.consultation_id == (queue_state.consultation_id or queue_state.id)),
        None,
    )

    triage_payload = preliminary_state.triage_result.model_dump() if preliminary_state.triage_result else None

    patient_name = _format_patient_name(patient)
    message = f"Patient {patient_name} checked in successfully"

    return StandardResponse(
        success=True,
        data={
            "queue_state": queue_state.model_dump(),
            "triage": triage_payload,
            "queue_item": queue_snapshot.model_dump() if queue_snapshot else None,
        },
        message=message,
    )


# Vitals payload mirrors frontend submission
class VitalsPayload(BaseModel):
    heart_rate: int = Field(..., ge=20, le=250)
    blood_pressure_systolic: int = Field(..., ge=50, le=300)
    blood_pressure_diastolic: int = Field(..., ge=20, le=200)
    respiratory_rate: int = Field(..., ge=5, le=80)
    temperature_celsius: float = Field(..., ge=25.0, le=45.0)
    oxygen_saturation: int = Field(..., ge=50, le=100)
    weight_kg: Optional[float] = Field(default=None, ge=2.0, le=400.0)


@router.post("/consultations/{consultation_id}/vitals", response_model=TriageResult)
def submit_vitals(
    consultation_id: str,
    vitals: VitalsPayload,
    session: Session = Depends(get_db_session),
    _: User = Depends(require_roles(UserRole.NURSE, UserRole.DOCTOR, UserRole.ADMIN)),
) -> TriageResult:
    """Submit vitals for a consultation and update triage/queue priority."""
    consultation = (
        session.query(Consultation)
        .filter(
            Consultation.id == consultation_id,
            Consultation.is_deleted.is_(False),
        )
        .first()
    )
    if consultation is None:
        raise HTTPException(status_code=404, detail="Consultation not found")

    recorded_at = datetime.now(timezone.utc)
    dto_vitals = VitalsSubmission(
        heart_rate=vitals.heart_rate,
        blood_pressure_systolic=vitals.blood_pressure_systolic,
        blood_pressure_diastolic=vitals.blood_pressure_diastolic,
        respiratory_rate=vitals.respiratory_rate,
        temperature_celsius=vitals.temperature_celsius,
        oxygen_saturation=vitals.oxygen_saturation,
        weight_kg=vitals.weight_kg,
    )

    vital_record = Vital(
        consultation_id=consultation_id,
        recorded_at=recorded_at,
        heart_rate=dto_vitals.heart_rate,
        respiratory_rate=dto_vitals.respiratory_rate,
        systolic_bp=dto_vitals.blood_pressure_systolic,
        diastolic_bp=dto_vitals.blood_pressure_diastolic,
        temperature=float(dto_vitals.temperature_celsius),
        oxygen_saturation=int(dto_vitals.oxygen_saturation),
        extra=({"weight_kg": dto_vitals.weight_kg} if dto_vitals.weight_kg is not None else None),
    )
    session.add(vital_record)

    state_result = state_machine.process_vitals_submission(consultation_id, dto_vitals)
    triage_result = (
        state_result.triage_result
        if state_result.triage_result
        else triage_engine.calculate_triage_level(
            vitals,
            consultation.chief_complaint or "",
        )
    )

    queue_state = (
        session.query(QueueState)
        .filter(
            QueueState.consultation_id == consultation_id,
            QueueState.is_deleted.is_(False),
        )
        .order_by(QueueState.created_at.desc())
        .first()
    )

    triage_level = triage_result.triage_level
    if queue_state:
        try:
            queue_service.transition_stage(
                queue_state_id=queue_state.id,
                next_stage=QueueStage.TRIAGE,
                notes="Vitals submitted",
                user_id=None,
                priority_level=triage_level,
            )
        except ValueError:
            with get_session() as write_session:
                persisted_state = crud.get_queue_state(write_session, queue_state.id)
                if persisted_state:
                    crud.update_queue_state(
                        write_session,
                        persisted_state,
                        priority_level=triage_level,
                    )

    consultation_updates: Dict[str, Any] = {"status": ConsultationStatus.TRIAGE}
    if state_result.assigned_doctor_id:
        consultation_updates["assigned_doctor_id"] = state_result.assigned_doctor_id
    consultation_updates["triage_level"] = triage_level
    crud.update_consultation(session, consultation, **consultation_updates)

    session.commit()
    return triage_result


@router.post(
    "/consultations/{consultation_id}/records",
    response_model=StandardResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_consultation_records(
    consultation_id: str,
    files: List[UploadFile] = File(...),
    notes: Optional[str] = Form(None),
    session: Session = Depends(get_db_session),
    current_user: User = Depends(require_roles(UserRole.NURSE, UserRole.DOCTOR, UserRole.ADMIN)),
) -> StandardResponse:
    if not files:
        raise HTTPException(status_code=400, detail="At least one file must be provided.")

    consultation = crud.get_consultation(session, consultation_id)
    if consultation is None:
        raise HTTPException(status_code=404, detail="Consultation not found")

    uploaded_records: List[ConsultationRecord] = []

    processing_results: List[Dict[str, Any]] = []

    for upload in files:
        record, _ = await storage_service.save_file_asset(
            upload,
            owner_type="consultation",
            owner_id=consultation_id,
            uploaded_by=str(current_user.id),
            description=notes,
        )
        processing_result = await document_processing_service.process_document(
            record.id,
            patient_id=consultation.patient_id,
            consultation_id=consultation_id,
        )
        processing_results.append(processing_result.to_dict())
        refreshed = crud.get_file_asset(session, record.id)
        if refreshed:
            uploaded_records.append(_serialize_consultation_record(session, refreshed))

    return StandardResponse(
        success=True,
        data={
            "records": [record.model_dump() for record in uploaded_records],
            "processing_results": processing_results,
        },
        message=(f"Uploaded {len(uploaded_records)} record(s); document processing completed."),
    )


@router.get(
    "/consultations/{consultation_id}/records",
    response_model=StandardResponse,
)
def list_consultation_records(
    consultation_id: str,
    session: Session = Depends(get_db_session),
    _: User = Depends(require_roles(UserRole.NURSE, UserRole.DOCTOR, UserRole.ADMIN)),
) -> StandardResponse:
    consultation = crud.get_consultation(session, consultation_id)
    if consultation is None:
        raise HTTPException(status_code=404, detail="Consultation not found")

    records = crud.list_file_assets(session, owner_type="consultation", owner_id=consultation_id)
    payload = [_serialize_consultation_record(session, record).model_dump() for record in records]
    return StandardResponse(success=True, data={"records": payload})


@router.get("/records/{file_id}/download")
def download_consultation_record(
    file_id: str,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_db_session),
    _: User = Depends(require_roles(UserRole.NURSE, UserRole.DOCTOR, UserRole.ADMIN)),
):
    record = crud.get_file_asset(session, file_id)
    if record is None or record.owner_type != "consultation":
        raise HTTPException(status_code=404, detail="Record not found")

    file_path = storage_service.resolve_file_asset_path(record)
    if not storage_service.is_local_backend:
        background_tasks.add_task(_safe_remove_file, file_path)

    filename = record.original_filename or f"consultation-record-{file_id}"
    media_type = record.content_type or "application/octet-stream"
    disposition = "inline"
    headers = {"Content-Disposition": f'{disposition}; filename="{filename}"'}

    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename,
        headers=headers,
    )


@router.get("/records/{file_id}/status", response_model=StandardResponse)
def get_consultation_record_status(
    file_id: str,
    session: Session = Depends(get_db_session),
    _: User = Depends(require_roles(UserRole.NURSE, UserRole.DOCTOR, UserRole.RECEPTIONIST, UserRole.ADMIN)),
) -> StandardResponse:
    record = crud.get_file_asset(session, file_id)
    if record is None or record.owner_type != "consultation":
        raise HTTPException(status_code=404, detail="Record not found")

    timeline_events = crud.list_timeline_events_for_file(session, record.id)
    status_value = (
        record.status.value
        if isinstance(record.status, DocumentProcessingStatus)
        else str(record.status or DocumentProcessingStatus.UPLOADED.value)
    )
    response = DocumentStatusResponse(
        file_id=record.id,
        status=status_value,
        processed_at=record.processed_at,
        confidence=float(record.confidence) if record.confidence is not None else None,
        needs_review=status_value
        in {
            DocumentProcessingStatus.NEEDS_REVIEW.value,
            DocumentProcessingStatus.FAILED.value,
        },
        processing_notes=record.processing_notes,
        timeline_event_ids=[event.id for event in timeline_events],
        metadata=record.processing_metadata or {},
    )
    return StandardResponse(success=True, data=response.model_dump())


@router.post("/records/{file_id}/review", response_model=StandardResponse)
def review_consultation_record(
    file_id: str,
    payload: DocumentReviewRequest,
    session: Session = Depends(get_db_session),
    current_user: User = Depends(require_roles(UserRole.NURSE, UserRole.DOCTOR, UserRole.ADMIN)),
) -> StandardResponse:
    record = crud.get_file_asset(session, file_id)
    if record is None or record.owner_type != "consultation":
        raise HTTPException(status_code=404, detail="Record not found")

    consultation = crud.get_consultation(session, record.owner_id)
    if consultation is None:
        raise HTTPException(status_code=404, detail="Consultation not found")

    status_map = {
        DocumentReviewResolution.APPROVED: DocumentProcessingStatus.COMPLETED,
        DocumentReviewResolution.NEEDS_REVIEW: DocumentProcessingStatus.NEEDS_REVIEW,
        DocumentReviewResolution.FAILED: DocumentProcessingStatus.FAILED,
    }
    new_status = status_map[payload.resolution]
    notes = payload.notes.strip() if payload.notes else None
    metadata = dict(record.processing_metadata or {})
    reviews = metadata.setdefault("reviews", [])
    reviews.append(
        {
            "reviewed_by": str(current_user.id),
            "reviewed_by_name": _format_user_name(current_user),
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "resolution": payload.resolution.value,
            "notes": notes,
        }
    )
    processed_at = datetime.now(timezone.utc)
    processing_notes = record.processing_notes or ""
    if notes:
        processing_notes = f"{processing_notes}\n{notes}" if processing_notes else notes

    updated_record = crud.update_file_asset(
        session,
        record,
        status=new_status,
        processing_metadata=metadata,
        processing_notes=processing_notes,
        processed_at=processed_at,
    )

    timeline_event_ids: List[str] = []
    if payload.update_timeline:
        timeline_events = crud.list_timeline_events_for_file(session, file_id)
        timeline_event_ids = [event.id for event in timeline_events]
        event_status = (
            TimelineEventStatus.FAILED
            if new_status == DocumentProcessingStatus.FAILED
            else (
                TimelineEventStatus.NEEDS_REVIEW
                if new_status == DocumentProcessingStatus.NEEDS_REVIEW
                else TimelineEventStatus.COMPLETED
            )
        )
        for event in timeline_events:
            new_notes = event.notes or ""
            if notes:
                review_line = f"Review note: {notes}"
                new_notes = f"{new_notes}\n{review_line}" if new_notes else review_line
            crud.update_timeline_event(
                session,
                event,
                status=event_status,
                notes=new_notes or event.notes,
            )
        session.flush()

    refreshed_payload = _serialize_consultation_record(session, updated_record)

    event_payload = {
        "file_id": updated_record.id,
        "patient_id": consultation.patient_id,
        "consultation_id": consultation.id,
        "status": new_status.value,
        "processed_at": (updated_record.processed_at or processed_at).isoformat(),
        "timeline_event_ids": timeline_event_ids,
        "confidence": (float(updated_record.confidence) if updated_record.confidence is not None else None),
        "needs_review": new_status
        in {
            DocumentProcessingStatus.NEEDS_REVIEW,
            DocumentProcessingStatus.FAILED,
        },
        "metadata": metadata,
        "reviewed_by": str(current_user.id),
        "resolution": payload.resolution.value,
        "notes": notes,
    }
    for channel in (
        f"documents:patient:{consultation.patient_id}",
        f"documents:consultation:{consultation.id}",
    ):
        notification_service.publish(channel, event_payload)

    return StandardResponse(
        success=True,
        data={"record": refreshed_payload.model_dump()},
        message="Document review updated.",
    )


@router.get("/consultations/{consultation_id}/records/stream")
async def stream_consultation_document_updates(
    consultation_id: str,
    _: User = Depends(require_roles(UserRole.NURSE, UserRole.DOCTOR, UserRole.ADMIN)),
) -> StreamingResponse:
    channel = f"documents:consultation:{consultation_id}"
    queue = await notification_service.subscribe(channel)

    async def event_generator():
        try:
            while True:
                event = await queue.get()
                if event.get("consultation_id") and event["consultation_id"] != consultation_id:
                    continue
                payload = json.dumps(event, ensure_ascii=False)
                yield f"data: {payload}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            await notification_service.unsubscribe(channel, queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/patients/{patient_id}/timeline", response_model=StandardResponse)
async def get_patient_timeline_summary(
    patient_id: str,
    force_refresh: bool = False,
    visit_limit: Optional[int] = None,
    _: User = Depends(require_roles(UserRole.DOCTOR, UserRole.NURSE, UserRole.ADMIN)),
) -> StandardResponse:
    try:
        result = await timeline_summary_service.generate_patient_summary(
            patient_id,
            force_refresh=force_refresh,
            visit_limit=visit_limit,
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Patient not found")

    response = TimelineSummaryResponse(
        summary_id=result["summary_id"],
        summary=result["summary"],
        timeline=result["timeline"],
        highlights=result.get("highlights", []),
        confidence=result.get("confidence"),
        cached=result.get("cached", False),
        generated_at=result.get("generated_at"),
        model=result.get("model"),
        token_usage=result.get("token_usage", {}),
    )
    return StandardResponse(success=True, data=response.model_dump())


@router.get("/queue", response_model=QueueResponse)
def get_queue(session: Session = Depends(get_db_session)) -> QueueResponse:
    """Return queue projections enhanced with ManageAgent triage and wait times."""
    items = _build_queue_items(session)
    total_wait = sum(item.wait_time_minutes for item in items)

    triage_distribution: Dict[int, int] = {}
    for item in items:
        level = item.triage_level or 3
        triage_distribution[level] = triage_distribution.get(level, 0) + 1

    average_wait = total_wait / len(items) if items else 0.0

    return QueueResponse(
        patients=items,
        total_count=len(items),
        average_wait_time=average_wait,
        triage_distribution=triage_distribution,
    )


@router.get("/queue/summary", response_model=dict)
def get_queue_summary(session: Session = Depends(get_db_session)) -> Dict[str, Any]:
    """Return aggregated queue status counts."""
    states = crud.list_queue_states(session, limit=500)

    totals: Dict[QueueStage, int] = {
        QueueStage.WAITING: 0,
        QueueStage.TRIAGE: 0,
        QueueStage.SCRIBE: 0,
        QueueStage.DISCHARGE: 0,
    }
    for state in states:
        totals[state.current_stage] = totals.get(state.current_stage, 0) + 1

        return {
            "success": True,
            "queue_summary": {
                "total_patients": sum(totals.values()),
                "awaiting_vitals": totals.get(QueueStage.WAITING, 0),
                "in_queue": totals.get(QueueStage.TRIAGE, 0),
                "assigned_to_doctor": totals.get(QueueStage.SCRIBE, 0),
                "in_consultation": totals.get(QueueStage.DISCHARGE, 0),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def _build_queue_items(session: Session) -> List[PatientQueueItem]:
    states = crud.list_queue_states(session, limit=200)
    items: List[PatientQueueItem] = []

    for state in states:
        item = _convert_state_to_queue_item(session, state)
        items.append(item)

    if not items:
        return items

    for item in items:
        prediction = wait_time_predictor.predict_wait_time(item, items)
        item.estimated_wait_minutes = prediction["estimated_wait_minutes"]
        item.queue_position = prediction["queue_position"]
        item.confidence_level = prediction["confidence_level"]
        item.prediction_method = prediction["prediction_method"]

    items.sort(key=lambda entry: entry.queue_position or len(items) + 1)
    return items


def _convert_state_to_queue_item(session: Session, state: QueueState) -> PatientQueueItem:
    patient = state.patient
    consultation = state.consultation
    assigned_doctor: Optional[User] = consultation.assigned_doctor if consultation else None

    wait_minutes = _minutes_since(state.created_at)
    triage_level = state.priority_level or 3

    vitals_data = _latest_vitals(session, consultation.id) if consultation else None

    item = PatientQueueItem(
        queue_state_id=str(state.id),
        consultation_id=consultation.id if consultation else state.id,
        patient_id=state.patient_id,
        patient_name=_format_patient_name(patient),
        age=_calculate_age(patient.date_of_birth) if patient else None,
        triage_level=triage_level,
        status=state.current_stage.value,
        wait_time_minutes=wait_minutes,
        priority_score=triage_engine.calculate_priority_score(triage_level, wait_minutes),
        assigned_doctor=_format_doctor(assigned_doctor),
        assigned_doctor_id=consultation.assigned_doctor_id if consultation else None,
        check_in_time=state.created_at,
        chief_complaint=consultation.chief_complaint if consultation else "",
        vitals=vitals_data,
    )
    return item


def _format_patient_name(patient: Optional[Patient]) -> str:
    if patient is None:
        return "Unknown patient"
    parts = [patient.first_name, patient.last_name]
    full_name = " ".join(part for part in parts if part).strip()
    if full_name:
        return full_name
    if patient.mrn:
        return patient.mrn
    return patient.id


def _format_doctor(user: Optional[User]) -> Optional[str]:
    if user is None:
        return None
    parts = [user.first_name, user.last_name]
    full_name = " ".join(part for part in parts if part).strip()
    if full_name:
        return f"Dr. {full_name}"
    return f"Dr. {user.email}"


def _format_user_name(user: Optional[User]) -> Optional[str]:
    if user is None:
        return None
    parts = [user.first_name, user.last_name]
    full_name = " ".join(part for part in parts if part).strip()
    if full_name:
        return full_name
    return user.email


def _record_download_url(record_id: str) -> str:
    return f"/api/v1/manage/records/{record_id}/download"


def _serialize_consultation_record(session: Session, record: FileAsset) -> ConsultationRecord:
    uploader = session.get(User, record.uploaded_by) if record.uploaded_by else None
    signed_url = storage_service.generate_file_asset_signed_url(record)
    timeline_events = crud.list_timeline_events_for_file(session, record.id)
    status_value = (
        record.status.value
        if isinstance(record.status, DocumentProcessingStatus)
        else str(record.status or DocumentProcessingStatus.UPLOADED.value)
    )
    needs_review = status_value in {
        DocumentProcessingStatus.NEEDS_REVIEW.value,
        DocumentProcessingStatus.FAILED.value,
    }
    metadata = record.processing_metadata or {}
    confidence = float(record.confidence) if record.confidence is not None else None
    return ConsultationRecord(
        id=record.id,
        original_filename=record.original_filename,
        content_type=record.content_type,
        size_bytes=record.size_bytes,
        description=record.description,
        uploaded_at=record.created_at,
        uploaded_by=_format_user_name(uploader),
        signed_url=signed_url,
        download_url=_record_download_url(record.id),
        status=status_value,
        document_type=record.document_type,
        confidence=confidence,
        needs_review=needs_review,
        processed_at=record.processed_at,
        processing_notes=record.processing_notes,
        processing_metadata=metadata,
        timeline_event_ids=[event.id for event in timeline_events],
    )


def _safe_remove_file(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _calculate_age(dob: Optional[date]) -> Optional[int]:
    if not dob:
        return None
    today = date.today()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    return age if age >= 0 else None


def _minutes_since(timestamp: Optional[datetime]) -> int:
    if timestamp is None:
        return 0
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    delta = now - timestamp
    return max(0, int(delta.total_seconds() // 60))


def _latest_vitals(session: Session, consultation_id: str) -> Optional[Dict[str, Any]]:
    vital = (
        session.query(Vital)
        .filter(
            Vital.consultation_id == consultation_id,
            Vital.is_deleted.is_(False),
        )
        .order_by(Vital.recorded_at.desc())
        .first()
    )
    if vital is None:
        return None

    return {
        "heart_rate": vital.heart_rate,
        "blood_pressure_systolic": vital.systolic_bp,
        "blood_pressure_diastolic": vital.diastolic_bp,
        "temperature": (float(vital.temperature) if vital.temperature is not None else None),
        "oxygen_saturation": vital.oxygen_saturation,
        "respiratory_rate": vital.respiratory_rate,
    }
