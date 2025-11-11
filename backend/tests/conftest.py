"""Shared pytest fixtures for backend tests."""

from __future__ import annotations

import os

# Set environment variable to indicate we're in a test environment
# This must be done BEFORE importing anything that might import the app
# so that services can detect the test environment during initialization
os.environ["MEDI_OS_TEST_ENV"] = "1"

# CRITICAL: Patch bcrypt.hashpw to handle >72 byte passwords gracefully
# This must happen before passlib.context.CryptContext is imported/initialized
# The issue: passlib tries to detect a wrap bug by testing with a >72 byte password
# which causes ValueError in newer bcrypt versions
try:
    import bcrypt as _bcrypt_lib

    # Store original hashpw
    _original_bcrypt_hashpw = _bcrypt_lib.hashpw

    def _safe_bcrypt_hashpw(secret, salt):
        """Safe bcrypt.hashpw that truncates >72 byte passwords."""
        # If secret is too long, truncate it to 72 bytes
        # This is what passlib does internally, but we need to do it earlier
        if isinstance(secret, bytes) and len(secret) > 72:
            secret = secret[:72]
        elif isinstance(secret, str):
            secret_bytes = secret.encode("utf-8")
            if len(secret_bytes) > 72:
                secret_bytes = secret_bytes[:72]
                # Try to decode, but if it fails, use the bytes directly
                try:
                    secret = secret_bytes.decode("utf-8", errors="ignore")
                except:
                    secret = secret_bytes
        try:
            return _original_bcrypt_hashpw(secret, salt)
        except ValueError as e:
            if "password cannot be longer than 72 bytes" in str(e):
                # Last resort: if it still fails, truncate more aggressively
                if isinstance(secret, bytes):
                    secret = secret[:72]
                else:
                    secret = secret.encode("utf-8")[:72].decode("utf-8", errors="ignore")
                return _original_bcrypt_hashpw(secret, salt)
            raise

    # Replace bcrypt.hashpw with our safe version
    _bcrypt_lib.hashpw = _safe_bcrypt_hashpw

    # Also patch passlib's detect_wrap_bug to return False
    import passlib.handlers.bcrypt as _bcrypt_mod

    if hasattr(_bcrypt_mod, "detect_wrap_bug"):

        @staticmethod
        def _patched_detect_wrap_bug(ident):
            """Patched version that skips the 72-byte password test."""
            return False  # Always return False to skip bug detection

        _bcrypt_mod.detect_wrap_bug = _patched_detect_wrap_bug
except (ImportError, AttributeError, Exception):
    # If patching fails, we'll handle errors in password hasher initialization
    pass

import contextlib
import sys
import types
from datetime import datetime, timezone
from typing import Generator
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# Now we can safely import the password hasher
from backend.security.password import password_hasher

# Stub out optional dependencies before importing app
# This allows tests to run even if these packages aren't installed
stub_whisper = types.ModuleType("whisper")
stub_whisper.load_model = lambda *args, **kwargs: None
sys.modules.setdefault("whisper", stub_whisper)

stub_google = types.ModuleType("google")
stub_generativeai = types.ModuleType("google.generativeai")
stub_generativeai.GenerativeModel = lambda *args, **kwargs: None
stub_generativeai.configure = lambda *args, **kwargs: None
stub_generativeai.list_models = lambda *args, **kwargs: []
sys.modules.setdefault("google", stub_google)
sys.modules.setdefault("google.generativeai", stub_generativeai)

# Stub langgraph
stub_langgraph = types.ModuleType("langgraph")
stub_langgraph_checkpoint = types.ModuleType("langgraph.checkpoint")
stub_langgraph_checkpoint_memory = types.ModuleType("langgraph.checkpoint.memory")


class StubMemorySaver:
    def __init__(self, *args, **kwargs):
        pass


stub_langgraph_checkpoint_memory.MemorySaver = StubMemorySaver

stub_langgraph_graph = types.ModuleType("langgraph.graph")


class StubStateGraph:
    def __init__(self, *args, **kwargs):
        self.nodes = {}
        self.edges = []

    def add_node(self, *args, **kwargs):
        pass

    def add_edge(self, *args, **kwargs):
        pass

    def set_entry_point(self, *args, **kwargs):
        pass

    def compile(self, *args, **kwargs):
        return self

    def invoke(self, *args, **kwargs):
        return {}

    def astream(self, *args, **kwargs):
        yield {}


stub_langgraph_graph.StateGraph = StubStateGraph
stub_langgraph_graph.END = "END"

sys.modules.setdefault("langgraph", stub_langgraph)
sys.modules.setdefault("langgraph.checkpoint", stub_langgraph_checkpoint)
sys.modules.setdefault("langgraph.checkpoint.memory", stub_langgraph_checkpoint_memory)
sys.modules.setdefault("langgraph.graph", stub_langgraph_graph)

# Import all models to ensure all tables are registered with Base.metadata
# This must happen before Base.metadata.create_all() is called
from backend.database import models  # noqa: F401 - Import entire module to register all models
from backend.database.base import Base
from backend.database.models import (
    Consultation,
    ConsultationStatus,
    Note,
    NoteVersion,
    Patient,
    QueueStage,
    QueueState,
    User,
)
from backend.main import app  # noqa: E402
from backend.security.password import password_hasher


@pytest.fixture(scope="function")
def db_session() -> Generator[Session, None, None]:
    """Create a test database session using SQLite in-memory.

    Uses SQLite for fast, isolated tests that don't require a database server.
    """
    from sqlalchemy.pool import StaticPool

    # Use SQLite in-memory database for tests
    # This is faster and doesn't require a database server
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Create all tables before creating session
    # Import models module to ensure all models are registered
    from backend.database import models  # noqa: F401

    Base.metadata.create_all(bind=engine)

    # Use autocommit=False but with explicit transaction management
    # This ensures we can control when transactions start/end
    TestingSessionLocal = sessionmaker(
        bind=engine, autocommit=False, autoflush=False, expire_on_commit=False
    )

    session = TestingSessionLocal()
    try:
        # Verify tables exist
        from sqlalchemy import inspect

        inspector = inspect(engine)
        tables = inspector.get_table_names()
        if not tables:
            # Force create tables if they don't exist
            Base.metadata.create_all(bind=engine)
            # Verify again
            tables = inspector.get_table_names()
            if not tables:
                raise RuntimeError(f"Failed to create tables. Expected tables but got: {tables}")

        # Ensure session is bound to the engine with tables
        assert session.bind is engine, "Session must be bound to the engine with tables"
        yield session
        # Rollback any uncommitted changes at the end of the test
        session.rollback()
    finally:
        session.close()
        # For SQLite in-memory, tables are automatically dropped when the connection closes
        # But we'll explicitly drop them for clarity
        try:
            Base.metadata.drop_all(bind=engine, checkfirst=True)
        except Exception:
            # Ignore errors during cleanup
            pass
        finally:
            engine.dispose()


@pytest.fixture(scope="function")
def client(db_session: Session) -> Generator[TestClient, None, None]:
    """Create a FastAPI test client with database session override."""
    from contextlib import contextmanager

    from backend.api.v1 import auth as auth_router
    from backend.database.session import get_db_session, get_session
    from backend.services.auth_service import AuthService

    # Ensure tables exist by binding the session to the engine that has tables
    # This ensures all queries use the same engine with tables
    engine = db_session.bind
    if engine:
        # Force create all tables on the engine if they don't exist
        Base.metadata.create_all(bind=engine)

    def override_get_db_session():
        # Ensure committed data is visible to API queries
        # For SQLite in-memory with StaticPool, all sessions share the same connection
        # The key is to ensure the session is in a clean, queryable state
        # SQLite requires explicit transaction management to see committed data
        try:
            # Commit any pending changes first to ensure they're persisted
            if db_session.in_transaction():
                db_session.commit()
        except Exception:
            # If commit fails, just pass - don't rollback as it would lose data
            pass
        # Expire all objects to force fresh queries from the database
        # This ensures queries will fetch from the database, not from session cache
        db_session.expire_all()
        # For SQLite with StaticPool, all sessions share the same connection
        # Committed data should be visible. We just need to ensure the session
        # is ready to query. SQLAlchemy will auto-begin a transaction when needed.
        try:
            yield db_session
        finally:
            # Don't auto-commit here - let the endpoint handle it if needed
            pass

    # Override get_session to use test database
    def override_get_session():
        try:
            yield db_session
            db_session.commit()
        except Exception:
            db_session.rollback()
            raise

    # Create an auth service that uses the test session
    @contextmanager
    def test_session_factory():
        try:
            yield db_session
            db_session.commit()
        except Exception:
            db_session.rollback()
            raise

    test_auth_service = AuthService(session_factory=test_session_factory)

    # Override the auth_service in the auth router to use the test session
    original_auth_service = auth_router.auth_service
    auth_router.auth_service = test_auth_service

    # Override both database session dependencies
    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_session] = override_get_session

    yield TestClient(app)

    # Restore original
    app.dependency_overrides.clear()
    auth_router.auth_service = original_auth_service


@pytest.fixture
def test_user(db_session: Session) -> User:
    """Create a test user."""
    from backend.database.models import Role, UserRole, UserStatus

    # Get or create a role first (since tables persist between tests in PostgreSQL)
    role = db_session.query(Role).filter(Role.name == "DOCTOR", Role.is_deleted == False).first()
    if not role:
        role = Role(
            id=str(uuid4()),
            name="DOCTOR",
            description="Doctor role",
            is_deleted=False,
        )
        db_session.add(role)
        db_session.flush()

    # Create user with unique email to avoid conflicts when tables persist between tests
    user_id = str(uuid4())
    unique_email = f"test-{user_id[:8]}@example.com"
    user = User(
        id=user_id,
        email=unique_email,
        password_hash=password_hasher.hash("testpassword123"),
        first_name="Test",
        last_name="User",
        status=UserStatus.ACTIVE,
        is_deleted=False,
    )
    db_session.add(user)
    db_session.flush()

    # Associate role with user through UserRole
    user_role = UserRole(
        user_id=user.id,
        role_id=role.id,
    )
    db_session.add(user_role)
    db_session.commit()
    db_session.refresh(user)
    db_session.refresh(role)
    return user


@pytest.fixture
def test_patient(db_session: Session) -> Patient:
    """Create a test patient."""
    patient = Patient(
        id=str(uuid4()),
        mrn=f"MRN-{uuid4().hex[:8]}",
        first_name="John",
        last_name="Doe",
        date_of_birth=datetime(1990, 1, 1, tzinfo=timezone.utc),
        sex="M",
        is_deleted=False,
    )
    db_session.add(patient)
    db_session.commit()
    db_session.refresh(patient)
    return patient


@pytest.fixture
def test_consultation(db_session: Session, test_patient: Patient, test_user: User) -> Consultation:
    """Create a test consultation."""
    consultation = Consultation(
        id=str(uuid4()),
        patient_id=test_patient.id,
        status=ConsultationStatus.IN_PROGRESS,
        chief_complaint="Test complaint",
        is_deleted=False,
    )
    db_session.add(consultation)
    db_session.commit()
    db_session.refresh(consultation)
    return consultation


@pytest.fixture
def test_queue_state(db_session: Session, test_consultation: Consultation) -> QueueState:
    """Create a test queue state."""
    queue_state = QueueState(
        id=str(uuid4()),
        consultation_id=test_consultation.id,
        stage=QueueStage.WAITING,
        is_deleted=False,
    )
    db_session.add(queue_state)
    db_session.commit()
    db_session.refresh(queue_state)
    return queue_state


@pytest.fixture
def test_note(db_session: Session, test_consultation: Consultation, test_user: User) -> Note:
    """Create a test note."""
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    note = Note(
        id=str(uuid4()),
        consultation_id=test_consultation.id,
        author_id=test_user.id,
        status="draft",
        is_deleted=False,
    )
    db_session.add(note)
    db_session.flush()  # Flush to get note.id

    note_version = NoteVersion(
        id=str(uuid4()),
        note_id=note.id,
        generated_by="ai",
        content="Test note content",
        is_ai_generated=True,
        created_by=test_user.id,
        is_deleted=False,
    )
    db_session.add(note_version)
    db_session.flush()  # Flush to ensure note_version is persisted before setting foreign key
    note.current_version_id = note_version.id
    db_session.flush()  # Flush again to ensure current_version_id is set
    db_session.commit()  # Commit to make data visible to other sessions

    # Ensure the note is visible to subsequent queries
    # Expire all objects to clear the session cache, then re-query
    db_session.expire_all()

    # Query the note again to ensure it's visible with all relationships
    # This ensures the note is actually in the database and can be found by API queries
    stmt = (
        select(Note)
        .where(Note.consultation_id == test_consultation.id, Note.is_deleted.is_(False))
        .options(selectinload(Note.current_version), selectinload(Note.versions))
    )
    queried_note = db_session.execute(stmt).scalars().first()

    # If note is None, something went wrong - raise an error
    if queried_note is None:
        raise RuntimeError(f"Failed to create or retrieve test note for consultation {test_consultation.id}")

    # Ensure the note is attached to the session and relationships are loaded
    # Refresh to ensure all relationships are properly loaded
    db_session.refresh(queried_note, ["current_version", "versions"])
    return queried_note


@pytest.fixture
def auth_headers(test_user: User, client: TestClient) -> dict[str, str]:
    """Get authentication headers for test user."""
    response = client.post(
        "/api/v1/auth/login",
        json={"email": test_user.email, "password": "testpassword123"},
    )
    assert response.status_code == 200
    data = response.json()
    token = data["data"]["access_token"]
    return {"Authorization": f"Bearer {token}"}
