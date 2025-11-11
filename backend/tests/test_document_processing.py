"""Tests for document processing service."""

from __future__ import annotations

from io import BytesIO
from uuid import uuid4

import pytest
from fastapi import UploadFile

from backend.database.models import Consultation, ConsultationStatus, DocumentProcessingStatus, FileAsset, Patient
from backend.database.session import get_session
from backend.services.document_processing import DocumentProcessingService


@pytest.fixture
def document_processing_service(db_session):
    """Create a document processing service with test database session."""
    from contextlib import contextmanager

    @contextmanager
    def get_session_override():
        yield db_session

    # Mock storage service
    class MockStorageService:
        async def save_file_asset(self, file: UploadFile, consultation_id: str, uploaded_by: str, **kwargs):
            file_asset = FileAsset(
                id=str(uuid4()),
                consultation_id=consultation_id,
                original_filename=file.filename or "test.pdf",
                storage_path=f"test/{file.filename}",
                mime_type=file.content_type or "application/pdf",
                size_bytes=100,
                uploaded_by=uploaded_by,
                is_deleted=False,
            )
            db_session.add(file_asset)
            db_session.commit()
            return file_asset, file_asset

    service = DocumentProcessingService(
        session_factory=get_session_override,
        storage_service=MockStorageService(),
    )
    return service


@pytest.mark.asyncio
async def test_process_document_success(document_processing_service, test_consultation, test_user, db_session):
    """Test successful document processing."""
    from backend.database import crud
    from backend.database.models import DocumentProcessingStatus

    # First create a file asset
    file_asset = crud.create_file_asset(
        db_session,
        owner_type="consultation",
        owner_id=test_consultation.id,
        original_filename="test.pdf",
        storage_path="test/test.pdf",
        content_type="application/pdf",
        size_bytes=100,
        uploaded_by=test_user.id,
        status=DocumentProcessingStatus.UPLOADED,
    )
    db_session.commit()

    result = await document_processing_service.process_document(
        file_asset.id,
        consultation_id=test_consultation.id,
    )

    assert result is not None
    assert result.success is True
    assert result.status in [DocumentProcessingStatus.COMPLETED, DocumentProcessingStatus.UPLOADED]


@pytest.mark.asyncio
async def test_process_document_creates_timeline_event(
    document_processing_service, test_consultation, test_user, test_patient, db_session
):
    """Test that document processing creates a timeline event."""
    from backend.database import crud
    from backend.database.models import DocumentProcessingStatus, TimelineEvent

    # First create a file asset
    file_asset = crud.create_file_asset(
        db_session,
        owner_type="consultation",
        owner_id=test_consultation.id,
        original_filename="test.pdf",
        storage_path="test/test.pdf",
        content_type="application/pdf",
        size_bytes=100,
        uploaded_by=test_user.id,
        status=DocumentProcessingStatus.UPLOADED,
    )
    db_session.commit()

    result = await document_processing_service.process_document(
        file_asset.id,
        consultation_id=test_consultation.id,
    )

    # Verify timeline event was created
    events = db_session.query(TimelineEvent).filter(TimelineEvent.consultation_id == test_consultation.id).all()
    assert len(events) > 0
    assert any(e.event_type.value == "document" for e in events)
