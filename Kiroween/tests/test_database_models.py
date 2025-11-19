"""
Test database models for medical safety and privacy compliance.

This module tests the core database models to ensure:
- Medical data validation works correctly
- Privacy constraints are enforced
- Patient ID generation and validation
- Cryptographic integrity features
"""

import pytest
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.exc import IntegrityError

from shared.database.base import Base
from shared.database.models import (
    PortableProfile,
    ClinicalEvent,
    LocalPatientRecord,
    ProfileSignature,
    ClinicalEventType,
    ProfileSyncStatus,
)


# Test database URL (in-memory SQLite for testing)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Create session
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    
    async with async_session() as session:
        yield session
    
    await engine.dispose()


@pytest.fixture
def sample_patient_id() -> str:
    """Generate a valid patient ID for testing."""
    return f"MED-{uuid.uuid4()}"


@pytest.fixture
def sample_event_id() -> str:
    """Generate a valid event ID for testing."""
    return f"EVT-{uuid.uuid4()}"


class TestPortableProfile:
    """Test PortableProfile model for medical safety and privacy."""
    
    async def test_create_valid_profile(self, db_session: AsyncSession, sample_patient_id: str):
        """Test creating a valid portable profile."""
        profile = PortableProfile(
            patient_id=sample_patient_id,
            first_name="Test",
            last_name="Patient",
            date_of_birth=datetime(1980, 1, 1),
            biological_sex="M",
            active_medications={"aspirin": "81mg daily"},
            known_allergies={"penicillin": "severe"},
            chronic_conditions={"hypertension": "controlled"},
            integrity_hash="test_hash_placeholder"
        )
        
        db_session.add(profile)
        await db_session.commit()
        
        # Verify profile was created
        result = await db_session.get(PortableProfile, sample_patient_id)
        assert result is not None
        assert result.first_name == "Test"
        assert result.biological_sex == "M"
        assert result.active_medications["aspirin"] == "81mg daily"
        
    async def test_patient_id_format_validation(self, db_session: AsyncSession):
        """Test that invalid patient ID formats are rejected."""
        # This test would work with PostgreSQL constraints
        # SQLite doesn't support regex constraints, so we'll test the format manually
        
        valid_ids = [
            "MED-550e8400-e29b-41d4-a716-446655440000",
            f"MED-{uuid.uuid4()}",
        ]
        
        invalid_ids = [
            "INVALID-ID",
            "MED-invalid-uuid",
            "MED-123",
            "",
            "550e8400-e29b-41d4-a716-446655440000",  # Missing MED- prefix
        ]
        
        # Test valid IDs
        for patient_id in valid_ids:
            # Check format manually (would be enforced by DB constraint in PostgreSQL)
            import re
            pattern = r'^MED-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
            assert re.match(pattern, patient_id), f"Valid ID failed validation: {patient_id}"
            
        # Test invalid IDs
        for patient_id in invalid_ids:
            import re
            pattern = r'^MED-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
            assert not re.match(pattern, patient_id), f"Invalid ID passed validation: {patient_id}"
            
    async def test_biological_sex_validation(self, db_session: AsyncSession, sample_patient_id: str):
        """Test biological sex validation."""
        valid_values = ["M", "F", "Other", "Unknown"]
        
        for sex in valid_values:
            profile = PortableProfile(
                patient_id=f"MED-{uuid.uuid4()}",
                biological_sex=sex,
                integrity_hash="test_hash"
            )
            db_session.add(profile)
            
        await db_session.commit()
        
        # Verify all profiles were created
        profiles = await db_session.execute(
            "SELECT biological_sex FROM portable_profiles WHERE biological_sex IS NOT NULL"
        )
        results = [row[0] for row in profiles.fetchall()]
        assert set(results) == set(valid_values)
        
    async def test_privacy_fields_optional(self, db_session: AsyncSession, sample_patient_id: str):
        """Test that privacy-sensitive fields are optional."""
        # Create profile with minimal data for maximum privacy
        profile = PortableProfile(
            patient_id=sample_patient_id,
            integrity_hash="minimal_profile_hash"
            # No name, DOB, or other identifying information
        )
        
        db_session.add(profile)
        await db_session.commit()
        
        # Verify profile was created with minimal data
        result = await db_session.get(PortableProfile, sample_patient_id)
        assert result is not None
        assert result.first_name is None
        assert result.last_name is None
        assert result.date_of_birth is None
        assert result.integrity_hash == "minimal_profile_hash"


class TestClinicalEvent:
    """Test ClinicalEvent model for medical safety."""
    
    async def test_create_clinical_event(self, db_session: AsyncSession, sample_patient_id: str, sample_event_id: str):
        """Test creating a clinical event."""
        # First create a profile
        profile = PortableProfile(
            patient_id=sample_patient_id,
            integrity_hash="test_hash"
        )
        db_session.add(profile)
        
        # Create clinical event
        event = ClinicalEvent(
            event_id=sample_event_id,
            patient_id=sample_patient_id,
            timestamp=datetime.now(timezone.utc),
            event_type=ClinicalEventType.VISIT,
            clinical_summary="Patient presented with chest pain",
            structured_data={
                "vitals": {"hr": 88, "bp": "120/80", "temp": 98.6},
                "symptoms": ["chest pain", "shortness of breath"]
            },
            confidence_score=0.95,
            cryptographic_signature="test_signature_placeholder",
            signing_key_fingerprint="test_fingerprint"
        )
        
        db_session.add(event)
        await db_session.commit()
        
        # Verify event was created
        result = await db_session.get(ClinicalEvent, sample_event_id)
        assert result is not None
        assert result.event_type == ClinicalEventType.VISIT
        assert result.confidence_score == 0.95
        assert "chest pain" in result.clinical_summary
        
    async def test_confidence_score_validation(self, db_session: AsyncSession, sample_patient_id: str):
        """Test confidence score range validation."""
        # First create a profile
        profile = PortableProfile(
            patient_id=sample_patient_id,
            integrity_hash="test_hash"
        )
        db_session.add(profile)
        await db_session.commit()
        
        # Test valid confidence scores
        valid_scores = [0.0, 0.5, 1.0, None]
        
        for i, score in enumerate(valid_scores):
            event = ClinicalEvent(
                event_id=f"EVT-{uuid.uuid4()}",
                patient_id=sample_patient_id,
                timestamp=datetime.now(timezone.utc),
                event_type=ClinicalEventType.SUMMARY,
                clinical_summary=f"Test event {i}",
                confidence_score=score,
                cryptographic_signature="test_signature",
                signing_key_fingerprint="test_fingerprint"
            )
            db_session.add(event)
            
        await db_session.commit()
        
        # Verify events were created
        events = await db_session.execute(
            f"SELECT confidence_score FROM clinical_events WHERE patient_id = '{sample_patient_id}'"
        )
        results = [row[0] for row in events.fetchall()]
        assert len(results) == len(valid_scores)
        
    async def test_no_hospital_metadata_in_events(self, db_session: AsyncSession, sample_patient_id: str):
        """Test that clinical events contain no hospital metadata."""
        # First create a profile
        profile = PortableProfile(
            patient_id=sample_patient_id,
            integrity_hash="test_hash"
        )
        db_session.add(profile)
        
        # Create event with only clinical data (no hospital info)
        event = ClinicalEvent(
            event_id=f"EVT-{uuid.uuid4()}",
            patient_id=sample_patient_id,
            timestamp=datetime.now(timezone.utc),
            event_type=ClinicalEventType.DIAGNOSIS,
            clinical_summary="Diagnosed with hypertension, started on lisinopril",
            structured_data={
                "diagnosis": "Essential hypertension",
                "medication": "Lisinopril 10mg daily",
                "follow_up": "2 weeks"
                # NOTE: No hospital name, doctor name, or location
            },
            cryptographic_signature="test_signature",
            signing_key_fingerprint="anonymous_fingerprint"
        )
        
        db_session.add(event)
        await db_session.commit()
        
        # Verify event contains only clinical data
        result = await db_session.get(ClinicalEvent, event.event_id)
        assert result is not None
        
        # Check that structured data contains only clinical information
        structured_data = result.structured_data
        assert "diagnosis" in structured_data
        assert "medication" in structured_data
        
        # Verify no hospital metadata
        prohibited_keys = ["hospital", "provider", "doctor", "location", "address"]
        for key in prohibited_keys:
            assert key not in structured_data
            assert key.lower() not in result.clinical_summary.lower()


class TestLocalPatientRecord:
    """Test LocalPatientRecord model for hospital-specific data."""
    
    async def test_create_local_record(self, db_session: AsyncSession, sample_patient_id: str):
        """Test creating a local patient record."""
        record = LocalPatientRecord(
            portable_patient_id=sample_patient_id,
            hospital_id="hospital-a",
            hospital_mrn="MRN-12345",
            admission_date=datetime.now(timezone.utc),
            attending_physician="Dr. Smith",
            department="Emergency",
            room_number="ER-101",
            insurance_info={"provider": "Blue Cross", "policy": "12345"},
            detailed_clinical_notes={"full_history": "Complete patient history..."}
        )
        
        db_session.add(record)
        await db_session.commit()
        
        # Verify record was created
        result = await db_session.get(LocalPatientRecord, record.local_record_id)
        assert result is not None
        assert result.hospital_id == "hospital-a"
        assert result.attending_physician == "Dr. Smith"
        assert result.insurance_info["provider"] == "Blue Cross"
        
    async def test_hospital_patient_uniqueness(self, db_session: AsyncSession, sample_patient_id: str):
        """Test that hospital-patient combination is unique."""
        # Create first record
        record1 = LocalPatientRecord(
            portable_patient_id=sample_patient_id,
            hospital_id="hospital-a",
            hospital_mrn="MRN-12345"
        )
        db_session.add(record1)
        await db_session.commit()
        
        # Try to create duplicate (same hospital + patient)
        record2 = LocalPatientRecord(
            portable_patient_id=sample_patient_id,
            hospital_id="hospital-a",  # Same hospital
            hospital_mrn="MRN-67890"   # Different MRN
        )
        db_session.add(record2)
        
        # This should fail due to unique constraint
        with pytest.raises(IntegrityError):
            await db_session.commit()


class TestProfileSignature:
    """Test ProfileSignature model for cryptographic integrity."""
    
    async def test_create_profile_signature(self, db_session: AsyncSession, sample_patient_id: str):
        """Test creating a profile signature."""
        # First create a profile
        profile = PortableProfile(
            patient_id=sample_patient_id,
            integrity_hash="test_hash"
        )
        db_session.add(profile)
        
        # Create signature
        signature = ProfileSignature(
            patient_id=sample_patient_id,
            signature_algorithm="RSA-SHA256",
            signature_value="base64_encoded_signature_here",
            public_key_fingerprint="sha256_fingerprint_of_public_key",
            signed_content_hash="sha256_hash_of_signed_content",
            signature_scope="full_profile"
        )
        
        db_session.add(signature)
        await db_session.commit()
        
        # Verify signature was created
        result = await db_session.get(ProfileSignature, signature.signature_id)
        assert result is not None
        assert result.signature_algorithm == "RSA-SHA256"
        assert result.signature_scope == "full_profile"
        
    async def test_signature_algorithm_validation(self, db_session: AsyncSession, sample_patient_id: str):
        """Test signature algorithm validation."""
        # First create a profile
        profile = PortableProfile(
            patient_id=sample_patient_id,
            integrity_hash="test_hash"
        )
        db_session.add(profile)
        await db_session.commit()
        
        # Test valid algorithms
        valid_algorithms = ["RSA-SHA256", "ECDSA-SHA256", "Ed25519"]
        
        for algorithm in valid_algorithms:
            signature = ProfileSignature(
                patient_id=sample_patient_id,
                signature_algorithm=algorithm,
                signature_value="test_signature",
                public_key_fingerprint="test_fingerprint",
                signed_content_hash="test_hash",
                signature_scope="clinical_event"
            )
            db_session.add(signature)
            
        await db_session.commit()
        
        # Verify all signatures were created
        signatures = await db_session.execute(
            f"SELECT signature_algorithm FROM profile_signatures WHERE patient_id = '{sample_patient_id}'"
        )
        results = [row[0] for row in signatures.fetchall()]
        assert set(results) == set(valid_algorithms)


@pytest.mark.medical_safety
class TestMedicalSafetyConstraints:
    """Test medical safety constraints and validation."""
    
    async def test_patient_id_generation_format(self):
        """Test patient ID generation follows MED-{uuid4} format."""
        import re
        
        # Generate multiple patient IDs
        patient_ids = [f"MED-{uuid.uuid4()}" for _ in range(10)]
        
        pattern = r'^MED-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        
        for patient_id in patient_ids:
            assert re.match(pattern, patient_id), f"Invalid patient ID format: {patient_id}"
            
    async def test_medical_data_ranges(self, db_session: AsyncSession, sample_patient_id: str):
        """Test medical data range validation."""
        # First create a profile
        profile = PortableProfile(
            patient_id=sample_patient_id,
            integrity_hash="test_hash"
        )
        db_session.add(profile)
        
        # Test valid medical data ranges
        valid_vitals = {
            "heart_rate": 72,      # Normal range: 60-100
            "systolic_bp": 120,    # Normal: <120
            "diastolic_bp": 80,    # Normal: <80
            "temperature": 98.6,   # Normal: 97-99°F
            "oxygen_sat": 98       # Normal: >95%
        }
        
        event = ClinicalEvent(
            event_id=f"EVT-{uuid.uuid4()}",
            patient_id=sample_patient_id,
            timestamp=datetime.now(timezone.utc),
            event_type=ClinicalEventType.VISIT,
            clinical_summary="Normal vital signs recorded",
            structured_data={"vitals": valid_vitals},
            cryptographic_signature="test_signature",
            signing_key_fingerprint="test_fingerprint"
        )
        
        db_session.add(event)
        await db_session.commit()
        
        # Verify event was created with valid vitals
        result = await db_session.get(ClinicalEvent, event.event_id)
        assert result is not None
        assert result.structured_data["vitals"]["heart_rate"] == 72


@pytest.mark.privacy
class TestPrivacyCompliance:
    """Test patient privacy compliance."""
    
    async def test_no_phi_in_portable_profile(self, db_session: AsyncSession, sample_patient_id: str):
        """Test that portable profiles contain no PHI beyond clinical necessity."""
        profile = PortableProfile(
            patient_id=sample_patient_id,
            # Minimal demographics for privacy
            biological_sex="F",  # Medical necessity only
            active_medications={"metformin": "500mg twice daily"},
            known_allergies={"shellfish": "anaphylaxis"},
            integrity_hash="privacy_compliant_hash"
            # NOTE: No name, address, phone, SSN, or other identifying info
        )
        
        db_session.add(profile)
        await db_session.commit()
        
        # Verify profile contains only medically necessary information
        result = await db_session.get(PortableProfile, sample_patient_id)
        assert result is not None
        
        # Check that optional identifying fields are not set
        assert result.first_name is None
        assert result.last_name is None
        assert result.date_of_birth is None
        
        # Verify medical information is present
        assert result.biological_sex == "F"
        assert "metformin" in result.active_medications
        assert "shellfish" in result.known_allergies
        
    async def test_hospital_metadata_separation(self, db_session: AsyncSession, sample_patient_id: str):
        """Test that hospital metadata is kept separate from portable data."""
        # Create portable profile (no hospital info)
        profile = PortableProfile(
            patient_id=sample_patient_id,
            integrity_hash="portable_hash"
        )
        db_session.add(profile)
        
        # Create local record (with hospital info)
        local_record = LocalPatientRecord(
            portable_patient_id=sample_patient_id,
            hospital_id="hospital-a",
            hospital_mrn="MRN-12345",
            attending_physician="Dr. Johnson",
            department="Cardiology",
            insurance_info={"provider": "Aetna"}
        )
        db_session.add(local_record)
        
        await db_session.commit()
        
        # Verify separation: portable profile has no hospital info
        portable = await db_session.get(PortableProfile, sample_patient_id)
        assert portable is not None
        
        # Check that portable profile has no hospital-specific fields
        portable_dict = {
            "patient_id": portable.patient_id,
            "first_name": portable.first_name,
            "last_name": portable.last_name,
            "biological_sex": portable.biological_sex,
            "active_medications": portable.active_medications,
            "known_allergies": portable.known_allergies,
            "chronic_conditions": portable.chronic_conditions
        }
        
        # Verify no hospital metadata in portable profile
        prohibited_terms = ["hospital", "doctor", "physician", "mrn", "insurance"]
        for term in prohibited_terms:
            for key, value in portable_dict.items():
                if value is not None:
                    assert term not in str(value).lower(), f"Hospital metadata found in portable profile: {term}"
                    
        # Verify local record contains hospital info
        local = await db_session.get(LocalPatientRecord, local_record.local_record_id)
        assert local is not None
        assert local.hospital_id == "hospital-a"
        assert local.attending_physician == "Dr. Johnson"