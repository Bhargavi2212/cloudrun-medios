"""Tests for CRUD operations."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from backend.database import crud
from backend.database.models import (Consultation, ConsultationStatus, Note,
                                     NoteVersion, Patient, QueueStage,
                                     QueueState, TimelineEvent,
                                     TimelineEventStatus, TimelineEventType,
                                     User)


def test_create_note_with_version(db_session, test_consultation, test_user):
    """Test creating a note with a version."""
    version = crud.create_note_with_version(
        db_session,
        consultation_id=test_consultation.id,
        note_content="Test note content",
        author_id=test_user.id,
        is_ai_generated=True,
        entities={"symptoms": ["fever", "cough"]},
    )

    assert version is not None
    version_id = version.id
    note_id = version.note_id

    # Verify note was created
    note = db_session.query(Note).filter(Note.id == note_id).first()
    assert note is not None
    assert note.consultation_id == test_consultation.id
    assert note.author_id == test_user.id
    assert note.status == "draft"
    assert note.current_version_id == version_id

    # Verify note version was created
    note_version = (
        db_session.query(NoteVersion).filter(NoteVersion.id == version_id).first()
    )
    assert note_version is not None
    assert note_version.content == "Test note content"
    assert note_version.is_ai_generated is True
    assert note_version.entities == {"symptoms": ["fever", "cough"]}


def test_get_note_for_consultation(db_session, test_note, test_consultation):
    """Test retrieving a note for a consultation."""
    note = crud.get_note_for_consultation(db_session, test_consultation.id)

    assert note is not None
    assert note.id == test_note.id
    assert note.consultation_id == test_consultation.id
    assert note.current_version is not None
    assert len(note.versions) > 0


def test_get_note_for_consultation_not_found(db_session):
    """Test retrieving a note for a non-existent consultation."""
    note = crud.get_note_for_consultation(db_session, str(uuid4()))
    assert note is None


def test_update_note_content(db_session, test_note, test_user):
    """Test updating note content."""
    new_content = "Updated note content"
    version = crud.update_note_content(
        db_session,
        consultation_id=test_note.consultation_id,
        content=new_content,
        author_id=test_user.id,
    )

    assert version is not None
    version_id = version.id

    # Verify new version was created
    note_version = (
        db_session.query(NoteVersion).filter(NoteVersion.id == version_id).first()
    )
    assert note_version is not None
    assert note_version.content == new_content
    assert note_version.created_by == test_user.id

    # Verify note status was reset to draft
    db_session.refresh(test_note)
    assert test_note.status == "draft"


def test_submit_note_for_approval(db_session, test_note):
    """Test submitting a note for approval."""
    note = crud.submit_note_for_approval(db_session, test_note.consultation_id)

    assert note is not None
    assert note.status == "pending_approval"


def test_approve_note(db_session, test_note, test_user):
    """Test approving a note."""
    # First submit for approval
    crud.submit_note_for_approval(db_session, test_note.consultation_id)

    # Then approve
    note = crud.approve_note(
        db_session, test_note.consultation_id, approver_id=test_user.id
    )

    assert note is not None
    assert note.status == "approved"


def test_reject_note(db_session, test_note, test_user):
    """Test rejecting a note."""
    # First submit for approval
    crud.submit_note_for_approval(db_session, test_note.consultation_id)

    # Then reject
    rejection_reason = "Incomplete information"
    note = crud.reject_note(
        db_session,
        test_note.consultation_id,
        rejection_reason=rejection_reason,
        approver_id=test_user.id,
    )

    assert note is not None
    assert note.status == "rejected"


def test_create_patient(db_session):
    """Test creating a patient."""
    from uuid import uuid4

    patient_data = {
        "mrn": f"MRN-{uuid4().hex[:8]}",
        "first_name": "Jane",
        "last_name": "Smith",
        "date_of_birth": datetime(1985, 5, 15, tzinfo=timezone.utc),
        "sex": "F",
        "contact_phone": "555-1234",
        "contact_email": "jane@example.com",
    }

    patient = crud.create_patient(db_session, **patient_data)

    assert patient is not None
    assert patient.first_name == "Jane"
    assert patient.last_name == "Smith"
    assert patient.sex == "F"
    assert patient.mrn is not None  # MRN should be auto-generated


def test_search_patients(db_session, test_patient):
    """Test searching for patients."""
    # Search by name
    results = crud.search_patients(db_session, query="John")
    assert len(results) > 0
    assert any(p.id == test_patient.id for p in results)

    # Search by MRN
    results = crud.search_patients(db_session, query=test_patient.mrn)
    assert len(results) > 0
    assert any(p.id == test_patient.id for p in results)


def test_list_timeline_events_for_patient(db_session, test_patient, test_consultation):
    """Test listing timeline events for a patient."""
    # Create a timeline event
    event = TimelineEvent(
        id=str(uuid4()),
        patient_id=test_patient.id,
        consultation_id=test_consultation.id,
        event_type=TimelineEventType.NOTE,
        event_date=datetime.now(timezone.utc),
        status=TimelineEventStatus.COMPLETED,
        title="Test event",
        summary="Test summary",
        is_deleted=False,
    )
    db_session.add(event)
    db_session.commit()

    # List events
    events = crud.list_timeline_events_for_patient(db_session, test_patient.id)
    assert len(events) > 0
    assert any(e.id == event.id for e in events)
