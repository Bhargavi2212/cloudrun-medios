"""Reusable CRUD helpers built on SQLAlchemy sessions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Optional, Sequence, Type, TypeVar
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from . import Base  # noqa: F401
from .models import (
    AudioFile,
    Consultation,
    ConsultationStatus,
    DocumentProcessingStatus,
    FileAsset,
    JobQueue,
    LLMUsage,
    Note,
    NoteVersion,
    Patient,
    PatientSummary,
    QueueEvent,
    QueueStage,
    QueueState,
    Role,
    ServiceMetric,
    SummaryCache,
    SummaryType,
    TimelineEvent,
    TimelineEventStatus,
    TimelineEventType,
    User,
)

ModelT = TypeVar("ModelT", bound=Base)


def get_user_by_email(session: Session, email: str) -> Optional[User]:
    stmt = select(User).where(User.email == email, User.is_deleted.is_(False))
    return session.execute(stmt).scalars().first()


def get_user_by_id(session: Session, user_id: UUID) -> Optional[User]:
    stmt = select(User).where(User.id == str(user_id), User.is_deleted.is_(False))
    return session.execute(stmt).scalars().first()


def list_patients(session: Session, skip: int = 0, limit: int = 50) -> Sequence[Patient]:
    stmt = (
        select(Patient)
        .where(Patient.is_deleted.is_(False))
        .order_by(Patient.last_name.asc(), Patient.first_name.asc())
        .offset(skip)
        .limit(limit)
    )
    return session.execute(stmt).scalars().all()


def count_patients(session: Session) -> int:
    return session.query(Patient).filter(Patient.is_deleted.is_(False)).count()


def search_patients(session: Session, query: str, limit: int = 25) -> Sequence[Patient]:
    pattern = f"%{query.lower()}%"
    stmt = (
        select(Patient)
        .where(
            Patient.is_deleted.is_(False),
            (Patient.first_name.ilike(pattern) | Patient.last_name.ilike(pattern) | Patient.mrn.ilike(pattern)),
        )
        .order_by(Patient.last_name.asc(), Patient.first_name.asc())
        .limit(limit)
    )
    return session.execute(stmt).scalars().all()


def create_patient(session: Session, **kwargs) -> Patient:
    return create_instance(session, Patient, **kwargs)


def create_consultation(session: Session, **kwargs) -> Consultation:
    return create_instance(session, Consultation, **kwargs)


def update_consultation(session: Session, consultation: Consultation, **kwargs) -> Consultation:
    for key, value in kwargs.items():
        setattr(consultation, key, value)
    session.add(consultation)
    session.flush()
    session.refresh(consultation)
    return consultation


def get_active_consultation_for_patient(session: Session, patient_id: str) -> Optional[Consultation]:
    active_statuses = (
        ConsultationStatus.INTAKE,
        ConsultationStatus.TRIAGE,
        ConsultationStatus.IN_PROGRESS,
    )
    stmt = (
        select(Consultation)
        .where(
            Consultation.patient_id == patient_id,
            Consultation.is_deleted.is_(False),
            Consultation.status.in_(active_statuses),
        )
        .order_by(Consultation.created_at.desc())
        .limit(1)
    )
    return session.execute(stmt).scalars().first()


def get_consultation(session: Session, consultation_id: str) -> Optional[Consultation]:
    stmt = select(Consultation).where(
        Consultation.id == str(consultation_id),
        Consultation.is_deleted.is_(False),
    )
    return session.execute(stmt).scalars().first()


def get_patient_by_mrn(session: Session, mrn: str) -> Optional[Patient]:
    stmt = select(Patient).where(Patient.mrn == mrn, Patient.is_deleted.is_(False))
    return session.execute(stmt).scalars().first()


def create_instance(session: Session, model: Type[ModelT], **kwargs) -> ModelT:
    instance = model(**kwargs)
    session.add(instance)
    session.flush()
    return instance


def soft_delete(session: Session, instance: ModelT) -> None:
    if hasattr(instance, "is_deleted"):
        instance.is_deleted = True  # type: ignore[attr-defined]
    session.add(instance)


def seed_roles(session: Session, roles: Iterable[dict]) -> None:
    existing = {name for (name,) in session.execute(select(Role.name))}
    for role_data in roles:
        if role_data["name"] in existing:
            continue
        create_instance(session, Role, **role_data)


def create_audio_file(session: Session, **kwargs) -> AudioFile:
    return create_instance(session, AudioFile, **kwargs)


def get_audio_file(session: Session, audio_id: str) -> Optional[AudioFile]:
    stmt = select(AudioFile).where(AudioFile.id == audio_id, AudioFile.is_deleted.is_(False))
    return session.execute(stmt).scalars().first()


def create_file_asset(session: Session, **kwargs) -> FileAsset:
    return create_instance(session, FileAsset, **kwargs)


def get_file_asset(session: Session, file_id: str) -> Optional[FileAsset]:
    stmt = select(FileAsset).where(FileAsset.id == file_id, FileAsset.is_deleted.is_(False))
    return session.execute(stmt).scalars().first()


def list_file_assets(
    session: Session,
    *,
    owner_type: str,
    owner_id: str,
) -> Sequence[FileAsset]:
    stmt = (
        select(FileAsset)
        .where(
            FileAsset.owner_type == owner_type,
            FileAsset.owner_id == owner_id,
            FileAsset.is_deleted.is_(False),
        )
        .order_by(FileAsset.created_at.desc())
    )
    return session.execute(stmt).scalars().all()


def update_file_asset(
    session: Session,
    asset: FileAsset,
    *,
    status: Optional[DocumentProcessingStatus] = None,
    confidence: Optional[float] = None,
    document_type: Optional[str] = None,
    processing_metadata: Optional[dict] = None,
    processing_notes: Optional[str] = None,
    processed_at: Optional[datetime] = None,
    last_error: Optional[str] = None,
) -> FileAsset:
    if status is not None:
        asset.status = status
    if confidence is not None:
        asset.confidence = confidence
    if document_type is not None:
        asset.document_type = document_type
    if processing_metadata is not None:
        asset.processing_metadata = processing_metadata
    if processing_notes is not None:
        asset.processing_notes = processing_notes
    if processed_at is not None:
        asset.processed_at = processed_at
    if last_error is not None:
        asset.last_error = last_error
    session.add(asset)
    session.flush()
    session.refresh(asset)
    return asset


def create_timeline_event(session: Session, **kwargs) -> TimelineEvent:
    return create_instance(session, TimelineEvent, **kwargs)


def get_timeline_event(session: Session, event_id: str) -> Optional[TimelineEvent]:
    stmt = select(TimelineEvent).where(TimelineEvent.id == event_id, TimelineEvent.is_deleted.is_(False))
    return session.execute(stmt).scalars().first()


def list_timeline_events_for_patient(
    session: Session,
    patient_id: str,
    *,
    include_pending: bool = False,
) -> Sequence[TimelineEvent]:
    stmt = (
        select(TimelineEvent)
        .where(
            TimelineEvent.patient_id == patient_id,
            TimelineEvent.is_deleted.is_(False),
        )
        .order_by(TimelineEvent.event_date.asc(), TimelineEvent.created_at.asc())
    )
    if not include_pending:
        # Explicitly include only COMPLETED and NEEDS_REVIEW events, exclude PENDING and FAILED
        stmt = stmt.where(
            TimelineEvent.status.in_(
                [
                    TimelineEventStatus.COMPLETED,
                    TimelineEventStatus.NEEDS_REVIEW,
                ]
            )
        )
    return session.execute(stmt).scalars().all()


def list_timeline_events_for_file(
    session: Session,
    file_id: str,
) -> Sequence[TimelineEvent]:
    stmt = (
        select(TimelineEvent)
        .where(
            TimelineEvent.source_file_id == file_id,
            TimelineEvent.is_deleted.is_(False),
        )
        .order_by(TimelineEvent.event_date.asc(), TimelineEvent.created_at.asc())
    )
    return session.execute(stmt).scalars().all()


def update_timeline_event(
    session: Session,
    event: TimelineEvent,
    *,
    status: Optional[TimelineEventStatus] = None,
    confidence: Optional[float] = None,
    summary: Optional[str] = None,
    data: Optional[dict] = None,
    notes: Optional[str] = None,
    event_type: Optional[TimelineEventType] = None,
    event_date: Optional[datetime] = None,
) -> TimelineEvent:
    if status is not None:
        event.status = status
    if confidence is not None:
        event.confidence = confidence
    if summary is not None:
        event.summary = summary
    if data is not None:
        event.data = data
    if notes is not None:
        event.notes = notes
    if event_type is not None:
        event.event_type = event_type
    if event_date is not None:
        event.event_date = event_date
    session.add(event)
    session.flush()
    session.refresh(event)
    return event


def create_patient_summary(
    session: Session,
    *,
    patient_id: str,
    summary_type: SummaryType,
    content: str,
    timeline: dict,
    llm_model: Optional[str] = None,
    token_usage: Optional[dict] = None,
    cost_cents: Optional[float] = None,
    created_by: Optional[str] = None,
) -> PatientSummary:
    summary = PatientSummary(
        patient_id=patient_id,
        summary_type=summary_type,
        content=content,
        timeline=timeline,
        llm_model=llm_model,
        token_usage=token_usage,
        cost_cents=cost_cents,
        created_by=created_by,
    )
    session.add(summary)
    session.flush()
    session.refresh(summary)
    return summary


def get_patient_summary(session: Session, summary_id: str) -> Optional[PatientSummary]:
    return session.get(PatientSummary, summary_id)


def get_summary_cache(session: Session, patient_id: str) -> Optional[SummaryCache]:
    return session.get(SummaryCache, patient_id)


def upsert_summary_cache(
    session: Session,
    patient_id: str,
    *,
    timeline_hash: str,
    summary_id: Optional[str],
    expires_at: Optional[datetime] = None,
) -> SummaryCache:
    cache = get_summary_cache(session, patient_id)
    now = datetime.now(timezone.utc)
    if cache is None:
        cache = SummaryCache(
            patient_id=patient_id,
            timeline_hash=timeline_hash,
            summary_id=summary_id,
            last_generated_at=now,
            expires_at=expires_at,
        )
        session.add(cache)
        session.flush()
        session.refresh(cache)
        return cache

    cache.timeline_hash = timeline_hash
    cache.summary_id = summary_id
    cache.last_generated_at = now
    cache.expires_at = expires_at
    session.add(cache)
    session.flush()
    session.refresh(cache)
    return cache


def create_job(
    session: Session,
    *,
    task_type: str,
    payload: dict,
    priority: int = 5,
    scheduled_at=None,
) -> JobQueue:
    data = {
        "task_type": task_type,
        "payload": payload,
        "priority": priority,
    }
    if scheduled_at is not None:
        data["scheduled_at"] = scheduled_at
    return create_instance(session, JobQueue, **data)


def get_job_by_id(session: Session, job_id: str) -> Optional[JobQueue]:
    stmt = select(JobQueue).where(JobQueue.id == job_id, JobQueue.is_deleted.is_(False))
    return session.execute(stmt).scalars().first()


def update_job(session: Session, job: JobQueue, **kwargs) -> JobQueue:
    for key, value in kwargs.items():
        setattr(job, key, value)
    session.add(job)
    session.flush()
    return job


def create_note_with_version(
    session: Session,
    *,
    consultation_id: str,
    author_id: Optional[str],
    note_content: str,
    entities: dict,
    is_ai_generated: bool,
) -> NoteVersion:
    import logging

    logger = logging.getLogger(__name__)

    logger.info(
        f"🔍 create_note_with_version: consultation_id={consultation_id}, content_length={len(note_content)}, is_ai_generated={is_ai_generated}"
    )

    # Check if note already exists
    note = session.query(Note).filter(Note.consultation_id == consultation_id, Note.is_deleted.is_(False)).first()

    if note is None:
        logger.info(f"📝 Creating new note for consultation {consultation_id}")
        note = create_instance(
            session,
            Note,
            consultation_id=consultation_id,
            author_id=author_id,
            status="draft",
        )
        session.flush()  # Ensure note.id is available
        logger.info(f"✅ Created new note: note_id={note.id}")
    else:
        logger.info(f"📝 Using existing note: note_id={note.id}, current_version_id={note.current_version_id}")

    # Create new version
    logger.info(f"📝 Creating new note version for note_id={note.id}")
    version = create_instance(
        session,
        NoteVersion,
        note_id=note.id,
        generated_by="ai" if is_ai_generated else "human",
        content=note_content,
        entities=entities,
        confidence=0.9 if is_ai_generated else None,
        is_ai_generated=is_ai_generated,
        created_by=author_id,
    )

    # Update note to point to new version
    note.current_version_id = version.id
    note.status = "draft"  # Reset to draft when updated
    session.add(note)
    session.flush()  # Ensure version.id is available
    session.refresh(version)
    session.refresh(note)

    logger.info(
        f"✅ Created note version: version_id={version.id}, note_id={note.id}, current_version_id={note.current_version_id}"
    )

    return version


def get_note_for_consultation(session: Session, consultation_id: str) -> Optional[Note]:
    """Get the note for a consultation, if it exists."""
    stmt = (
        select(Note)
        .where(Note.consultation_id == consultation_id, Note.is_deleted.is_(False))
        .options(selectinload(Note.current_version), selectinload(Note.versions))
    )
    return session.execute(stmt).scalars().first()


def update_note_content(
    session: Session,
    consultation_id: str,
    content: str,
    author_id: Optional[str],
) -> NoteVersion:
    """Update note content by creating a new version."""
    note = get_note_for_consultation(session, consultation_id)
    if not note:
        raise ValueError(f"No note found for consultation {consultation_id}")

    version = create_instance(
        session,
        NoteVersion,
        note_id=note.id,
        generated_by="human",
        content=content,
        entities=None,
        confidence=None,
        is_ai_generated=False,
        created_by=author_id,
    )

    note.current_version_id = version.id
    note.status = "draft"  # Reset to draft when updated
    session.add(note)
    session.flush()
    session.refresh(version)
    return version


def approve_note(
    session: Session,
    consultation_id: str,
    approver_id: Optional[str] = None,
) -> Note:
    """Approve a note, changing its status to 'approved'."""
    note = get_note_for_consultation(session, consultation_id)
    if not note:
        raise ValueError(f"No note found for consultation {consultation_id}")

    if note.status == "approved":
        return note  # Already approved

    note.status = "approved"
    session.add(note)
    session.flush()
    session.refresh(note)
    return note


def reject_note(
    session: Session,
    consultation_id: str,
    rejection_reason: Optional[str] = None,
    approver_id: Optional[str] = None,
) -> Note:
    """Reject a note, changing its status to 'rejected'."""
    note = get_note_for_consultation(session, consultation_id)
    if not note:
        raise ValueError(f"No note found for consultation {consultation_id}")

    note.status = "rejected"
    # Store rejection reason in processing_notes if available (or we could add a rejection_reason field)
    if note.current_version:
        # We could add a rejection_reason field to NoteVersion if needed
        pass
    session.add(note)
    session.flush()
    session.refresh(note)
    return note


def submit_note_for_approval(
    session: Session,
    consultation_id: str,
    author_id: Optional[str] = None,
) -> Note:
    """Submit a note for approval, changing its status to 'pending_approval'."""
    note = get_note_for_consultation(session, consultation_id)
    if not note:
        raise ValueError(f"No note found for consultation {consultation_id}")

    if not note.current_version or not note.current_version.content:
        raise ValueError("Note has no content to submit for approval")

    note.status = "pending_approval"
    session.add(note)
    session.flush()
    session.refresh(note)
    return note


def log_llm_usage(
    session: Session,
    *,
    request_id: Optional[str],
    user_id: Optional[str],
    model: str,
    tokens_prompt: int,
    tokens_completion: int,
    cost_cents: float,
    status: str,
) -> LLMUsage:
    return create_instance(
        session,
        LLMUsage,
        request_id=request_id,
        user_id=user_id,
        model=model,
        tokens_prompt=tokens_prompt,
        tokens_completion=tokens_completion,
        cost_cents=cost_cents,
        status=status,
    )


def log_service_metric(
    session: Session,
    *,
    service_name: str,
    metric_name: str,
    metric_value: float,
    metadata: Optional[dict] = None,
) -> ServiceMetric:
    return create_instance(
        session,
        ServiceMetric,
        service_name=service_name,
        metric_name=metric_name,
        metric_value=metric_value,
        metadata=metadata or {},
    )


# --------------------------------------------------------------------------- #
# Queue helpers
# --------------------------------------------------------------------------- #


def create_queue_state(
    session: Session,
    *,
    patient_id: str,
    consultation_id: Optional[str],
    stage: QueueStage,
    priority_level: int,
    estimated_wait_seconds: Optional[int] = None,
    assigned_to: Optional[str] = None,
) -> QueueState:
    return create_instance(
        session,
        QueueState,
        patient_id=patient_id,
        consultation_id=consultation_id,
        current_stage=stage,
        priority_level=priority_level,
        estimated_wait_seconds=estimated_wait_seconds,
        assigned_to=assigned_to,
    )


def get_queue_state(session: Session, queue_state_id: str) -> Optional[QueueState]:
    stmt = (
        select(QueueState)
        .where(
            QueueState.id == queue_state_id,
            QueueState.is_deleted.is_(False),
        )
        .options(
            selectinload(QueueState.patient),
            selectinload(QueueState.consultation),
        )
    )
    return session.execute(stmt).scalars().first()


def list_queue_states(
    session: Session,
    *,
    stage: Optional[QueueStage] = None,
    limit: int = 200,
) -> Sequence[QueueState]:
    stmt = (
        select(QueueState)
        .where(QueueState.is_deleted.is_(False))
        .options(
            selectinload(QueueState.patient),
            selectinload(QueueState.consultation),
        )
    )
    if stage is not None:
        stmt = stmt.where(QueueState.current_stage == stage)
    stmt = stmt.order_by(QueueState.priority_level.asc(), QueueState.created_at.asc()).limit(limit)
    return session.execute(stmt).scalars().all()


def update_queue_state(session: Session, queue_state: QueueState, **updates) -> QueueState:
    for key, value in updates.items():
        setattr(queue_state, key, value)
    session.add(queue_state)
    session.flush()
    session.refresh(queue_state)
    return queue_state


def log_queue_event(
    session: Session,
    *,
    queue_state_id: str,
    event_type: str,
    previous_stage: Optional[QueueStage],
    next_stage: Optional[QueueStage],
    notes: Optional[str],
    created_by: Optional[str],
) -> QueueEvent:
    return create_instance(
        session,
        QueueEvent,
        queue_state_id=queue_state_id,
        event_type=event_type,
        previous_stage=previous_stage,
        next_stage=next_stage,
        notes=notes,
        created_by=created_by,
    )
