"""
Unit tests for database repositories.
"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

from shared.database import (
    Base,
    ProfileRepository,
    ClinicalEventRepository,
    LocalRecordRepository,
    ClinicalEventType,
    ProfileSyncStatus
)


@pytest_asyncio.fixture
async def async_session():
    """Create async test database session."""
    # Use in-memory SQLite for testing
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=False
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Create session factory
    async_session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    async with async_session_factory() as session:
        yield session
    
    await engine.dispose()


@pytest.fixture
def sample_patient_id():
    """Sample patient ID for testing."""
    return "MED-550e8400-e29b-41d4-a716-446655440000"


class TestProfileRepository:
    """Test cases for ProfileRepository."""
    
    @pytest.mark.asyncio
    async def test_create_profile(self, async_session: AsyncSession, sample_patient_id: str):
        """Test creating a new patient profile."""
        repo = ProfileRepository(async_session)
        
        profile = await repo.create_profile(
            patient_id=sample_patient_id,
            first_name="John",
            last_name="Doe",
            date_of_birth=datetime(1980, 1, 1),
            biological_sex="M",
            active_medications={"medications": [{"name": "aspirin", "dose": "81mg"}]},
            known_allergies={"allergies": [{"allergen": "penicillin"}]},
            chronic_conditions={"conditions": [{"condition": "hypertension"}]}
        )
        
        assert profile.patient_id == sample_patient_id
        assert profile.first_name == "John"
        assert profile.last_name == "Doe"
        assert profile.biological_sex == "M"
        assert profile.active_medications["medications"][0]["name"] == "aspirin"
    
    @pytest.mark.asyncio
    async def test_get_profile_by_id(self, async_session: AsyncSession, sample_patient_id: str):
        """Test retrieving profile by ID."""
        repo = ProfileRepository(async_session)
        
        # Create profile
        await repo.create_profile(
            patient_id=sample_patient_id,
            first_name="Jane",
            last_name="Smith"
        )
        
        # Retrieve profile
        profile = await repo.get_by_id(sample_patient_id)
        
        assert profile is not None
        assert profile.patient_id == sample_patient_id
        assert profile.first_name == "Jane"
        assert profile.last_name == "Smith"
    
    async def test_search_profiles(self, async_session: AsyncSession):
        """Test searching profiles by demographics."""
        repo = ProfileRepository(async_session)
        
        # Create test profiles
        await repo.create_profile(
            patient_id="MED-111111111-1111-1111-1111-111111111111",
            first_name="Alice",
            last_name="Johnson"
        )
        await repo.create_profile(
            patient_id="MED-222222222-2222-2222-2222-222222222222",
            first_name="Bob",
            last_name="Johnson"
        )
        
        # Search by last name
        profiles = await repo.search_profiles(last_name="Johnson")
        
        assert len(profiles) == 2
        assert all(p.last_name == "Johnson" for p in profiles)
    
    async def test_update_medical_data(self, async_session: AsyncSession, sample_patient_id: str):
        """Test updating medical data."""
        repo = ProfileRepository(async_session)
        
        # Create profile
        await repo.create_profile(
            patient_id=sample_patient_id,
            first_name="Test",
            active_medications={"medications": []}
        )
        
        # Update medical data
        updated_profile = await repo.update_medical_data(
            patient_id=sample_patient_id,
            active_medications={"medications": [{"name": "lisinopril", "dose": "10mg"}]},
            known_allergies={"allergies": [{"allergen": "shellfish"}]}
        )
        
        assert updated_profile is not None
        assert updated_profile.active_medications["medications"][0]["name"] == "lisinopril"
        assert updated_profile.known_allergies["allergies"][0]["allergen"] == "shellfish"


class TestClinicalEventRepository:
    """Test cases for ClinicalEventRepository."""
    
    async def test_create_clinical_event(self, async_session: AsyncSession, sample_patient_id: str):
        """Test creating a clinical event."""
        repo = ClinicalEventRepository(async_session)
        
        event = await repo.create_clinical_event(
            patient_id=sample_patient_id,
            event_type=ClinicalEventType.VISIT,
            clinical_summary="Patient presented with chest pain",
            structured_data={"vitals": {"bp": "120/80", "hr": 72}},
            ai_generated_insights="Low risk chest pain",
            confidence_score=0.85
        )
        
        assert event.patient_id == sample_patient_id
        assert event.event_type == ClinicalEventType.VISIT
        assert event.clinical_summary == "Patient presented with chest pain"
        assert event.structured_data["vitals"]["bp"] == "120/80"
        assert event.confidence_score == 0.85
    
    async def test_get_patient_timeline(self, async_session: AsyncSession, sample_patient_id: str):
        """Test retrieving patient timeline."""
        repo = ClinicalEventRepository(async_session)
        
        # Create multiple events
        await repo.create_clinical_event(
            patient_id=sample_patient_id,
            event_type=ClinicalEventType.VISIT,
            clinical_summary="First visit",
            timestamp=datetime.utcnow() - timedelta(days=2)
        )
        await repo.create_clinical_event(
            patient_id=sample_patient_id,
            event_type=ClinicalEventType.LAB_RESULT,
            clinical_summary="Lab results",
            timestamp=datetime.utcnow() - timedelta(days=1)
        )
        
        # Get timeline
        timeline = await repo.get_patient_timeline(patient_id=sample_patient_id)
        
        assert len(timeline) == 2
        # Should be ordered by timestamp descending (most recent first)
        assert timeline[0].clinical_summary == "Lab results"
        assert timeline[1].clinical_summary == "First visit"
    
    async def test_get_timeline_summary(self, async_session: AsyncSession, sample_patient_id: str):
        """Test getting timeline summary statistics."""
        repo = ClinicalEventRepository(async_session)
        
        # Create events of different types
        await repo.create_clinical_event(
            patient_id=sample_patient_id,
            event_type=ClinicalEventType.VISIT,
            clinical_summary="Visit 1"
        )
        await repo.create_clinical_event(
            patient_id=sample_patient_id,
            event_type=ClinicalEventType.VISIT,
            clinical_summary="Visit 2"
        )
        await repo.create_clinical_event(
            patient_id=sample_patient_id,
            event_type=ClinicalEventType.LAB_RESULT,
            clinical_summary="Lab test"
        )
        
        # Get summary
        summary = await repo.get_timeline_summary(patient_id=sample_patient_id)
        
        assert summary["patient_id"] == sample_patient_id
        assert summary["total_events"] == 3
        assert summary["event_counts"]["visit"] == 2
        assert summary["event_counts"]["lab_result"] == 1
    
    async def test_search_events_by_content(self, async_session: AsyncSession, sample_patient_id: str):
        """Test searching events by clinical content."""
        repo = ClinicalEventRepository(async_session)
        
        # Create events with different content
        await repo.create_clinical_event(
            patient_id=sample_patient_id,
            event_type=ClinicalEventType.VISIT,
            clinical_summary="Patient has chest pain and shortness of breath"
        )
        await repo.create_clinical_event(
            patient_id=sample_patient_id,
            event_type=ClinicalEventType.VISIT,
            clinical_summary="Patient reports headache and nausea"
        )
        
        # Search for chest pain
        events = await repo.search_events_by_content(
            patient_id=sample_patient_id,
            search_term="chest pain"
        )
        
        assert len(events) == 1
        assert "chest pain" in events[0].clinical_summary


class TestLocalRecordRepository:
    """Test cases for LocalRecordRepository."""
    
    async def test_create_local_record(self, async_session: AsyncSession, sample_patient_id: str):
        """Test creating a local patient record."""
        repo = LocalRecordRepository(async_session)
        
        record = await repo.create_local_record(
            portable_patient_id=sample_patient_id,
            hospital_id="hospital-a",
            hospital_mrn="MRN-12345",
            attending_physician="Dr. Smith",
            department="Emergency",
            insurance_info={"provider": "Test Insurance", "policy": "POL-123"}
        )
        
        assert record.portable_patient_id == sample_patient_id
        assert record.hospital_id == "hospital-a"
        assert record.hospital_mrn == "MRN-12345"
        assert record.attending_physician == "Dr. Smith"
        assert record.department == "Emergency"
        assert record.insurance_info["provider"] == "Test Insurance"
    
    async def test_get_by_portable_id(self, async_session: AsyncSession, sample_patient_id: str):
        """Test retrieving local record by portable patient ID."""
        repo = LocalRecordRepository(async_session)
        
        # Create record
        await repo.create_local_record(
            portable_patient_id=sample_patient_id,
            hospital_id="hospital-a",
            hospital_mrn="MRN-67890"
        )
        
        # Retrieve record
        record = await repo.get_by_portable_id(
            portable_patient_id=sample_patient_id,
            hospital_id="hospital-a"
        )
        
        assert record is not None
        assert record.portable_patient_id == sample_patient_id
        assert record.hospital_mrn == "MRN-67890"
    
    async def test_get_hospital_patients(self, async_session: AsyncSession):
        """Test getting all patients for a hospital."""
        repo = LocalRecordRepository(async_session)
        
        # Create records for different hospitals
        await repo.create_local_record(
            portable_patient_id="MED-111111111-1111-1111-1111-111111111111",
            hospital_id="hospital-a",
            hospital_mrn="MRN-001"
        )
        await repo.create_local_record(
            portable_patient_id="MED-222222222-2222-2222-2222-222222222222",
            hospital_id="hospital-a",
            hospital_mrn="MRN-002"
        )
        await repo.create_local_record(
            portable_patient_id="MED-333333333-3333-3333-3333-333333333333",
            hospital_id="hospital-b",
            hospital_mrn="MRN-003"
        )
        
        # Get patients for hospital-a
        patients = await repo.get_hospital_patients(hospital_id="hospital-a")
        
        assert len(patients) == 2
        assert all(p.hospital_id == "hospital-a" for p in patients)
    
    async def test_update_sync_status(self, async_session: AsyncSession, sample_patient_id: str):
        """Test updating profile synchronization status."""
        repo = LocalRecordRepository(async_session)
        
        # Create record
        record = await repo.create_local_record(
            portable_patient_id=sample_patient_id,
            hospital_id="hospital-a",
            profile_sync_status=ProfileSyncStatus.PENDING
        )
        
        # Update sync status
        updated_record = await repo.update_sync_status(
            local_record_id=str(record.local_record_id),
            sync_status=ProfileSyncStatus.SYNCED,
            last_export=datetime.utcnow()
        )
        
        assert updated_record is not None
        assert updated_record.profile_sync_status == ProfileSyncStatus.SYNCED
        assert updated_record.last_profile_export is not None


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])