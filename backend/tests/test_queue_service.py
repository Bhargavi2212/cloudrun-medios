from __future__ import annotations

import contextlib
import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import crud
from backend.database.base import Base
from backend.database.models import (Consultation, Patient, QueueStage,
                                     QueueState)
from backend.services import queue_service as queue_module
from backend.services.queue_service import QueueNotifier, QueueService


class DummyNotifier(QueueNotifier):
    def __init__(self) -> None:
        super().__init__()
        self.messages = []

    async def broadcast(self, message):
        self.messages.append(message)


@pytest.fixture()
def sqlite_session(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(
        bind=engine, autocommit=False, autoflush=False, expire_on_commit=False
    )

    @contextlib.contextmanager
    def get_session_override():
        session = TestingSessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr(queue_module, "get_session", get_session_override)
    return TestingSessionLocal


@pytest.fixture()
def queue_service(monkeypatch, sqlite_session):
    notifier = DummyNotifier()
    service = QueueService(notifier=notifier)
    monkeypatch.setattr(queue_module, "queue_service", service, raising=False)
    monkeypatch.setattr(crud, "log_queue_event", lambda session, **kwargs: None)
    return service


def test_queue_creation_and_transition(queue_service, sqlite_session):
    # bootstrap patient record
    session = sqlite_session()
    patient = crud.create_instance(
        session,
        Patient,
        first_name="Test",
        last_name="Patient",
        date_of_birth=datetime.date(1990, 1, 1),
        mrn="MRN123",
    )
    session.commit()
    patient_id = patient.id
    session.close()

    created = queue_service.create_entry(
        patient_id=patient_id,
        consultation_id=None,
        chief_complaint="Headache",
        priority_level=2,
        estimated_wait_seconds=600,
        assigned_to=None,
        created_by=None,
    )
    assert created.current_stage == QueueStage.WAITING

    session_check = sqlite_session()
    assert session_check.query(QueueState).count() == 1
    consultation = session_check.query(Consultation).first()
    assert consultation is not None
    assert consultation.chief_complaint == "Headache"
    session_check.close()

    advanced = queue_service.transition_stage(
        created.id,
        next_stage=QueueStage.TRIAGE,
        notes="Vitals captured",
        user_id="nurse-1",
    )
    assert advanced.current_stage == QueueStage.TRIAGE

    snapshot = queue_service.snapshot()
    assert snapshot.totals_by_stage["triage"] == 1


def test_invalid_transition_raises(queue_service, sqlite_session):
    session = sqlite_session()
    patient = crud.create_instance(
        session,
        Patient,
        first_name="T",
        last_name="P",
        date_of_birth=datetime.date(1990, 1, 1),
        mrn="MRN999",
    )
    session.commit()
    patient_id = patient.id
    session.close()

    created = queue_service.create_entry(
        patient_id=patient_id,
        consultation_id=None,
        chief_complaint="Dizziness",
        priority_level=3,
        estimated_wait_seconds=300,
        assigned_to=None,
        created_by=None,
    )

    with pytest.raises(ValueError):
        queue_service.transition_stage(
            created.id,
            next_stage=QueueStage.SCRIBE,
            notes=None,
            user_id=None,
        )
