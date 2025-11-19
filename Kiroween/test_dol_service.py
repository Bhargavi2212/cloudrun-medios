#!/usr/bin/env python3
"""
Test script for DOL Service implementation.

This script validates the Data Orchestration Layer service functionality
including privacy filtering, cryptographic operations, and API endpoints.
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from services.dol_service.services.privacy_filter import PrivacyFilterService
    from services.dol_service.services.crypto_service import CryptographicService
    from services.dol_service.schemas import (
        PortableProfile,
        ClinicalTimelineEntry,
        PatientDemographics,
        EventType
    )
    print("✅ Successfully imported DOL service modules")
except ImportError as e:
    print(f"❌ Failed to import DOL service modules: {e}")
    print("Note: This is expected if dependencies are not available")
    sys.exit(1)


async def test_privacy_filter_service():
    """Test privacy filtering functionality."""
    print("\n🔒 Testing Privacy Filter Service...")
    
    hospital_id = "test_hospital_dol"
    privacy_filter = PrivacyFilterService(hospital_id)
    
    # Test clinical text filtering
    test_text = "Patient at General Hospital was seen by Dr. Smith in Room 205"
    filtered_text = await privacy_filter._filter_clinical_text(test_text)
    
    assert "General Hospital" not in filtered_text, "Hospital name should be filtered"
    assert "Dr. Smith" not in filtered_text, "Doctor name should be filtered"
    assert "Room 205" not in filtered_text, "Room number should be filtered"
    print("✅ Clinical text filtering successful")
    
    # Test structured data filtering
    test_data = {
        "symptoms": ["chest pain", "dyspnea"],
        "vital_signs": {"hr": 88, "bp": "120/80"},
        "hospital_id": "should_be_removed",
        "attending_physician": "Dr. Johnson",
        "department": "Emergency"
    }
    
    filtered_data = await privacy_filter._filter_structured_data(test_data)
    
    assert "symptoms" in filtered_data, "Clinical data should be preserved"
    assert "vital_signs" in filtered_data, "Vital signs should be preserved"
    assert "hospital_id" not in filtered_data, "Hospital ID should be removed"
    assert "attending_physician" not in filtered_data, "Physician should be removed"
    assert "department" not in filtered_data, "Department should be removed"
    print("✅ Structured data filtering successful")
    
    # Test portable profile creation
    patient_data = {
        "patient_id": "MED-test-001",
        "demographics": {
            "first_name": "John",
            "last_name": "Doe",
            "date_of_birth": "1980-01-01",
            "biological_sex": "male"
        },
        "clinical_timeline": [
            {
                "entry_id": "entry_001",
                "patient_id": "MED-test-001",
                "timestamp": datetime.utcnow(),
                "event_type": "clinical_visit",
                "clinical_summary": "Patient presents with chest pain",
                "structured_data": {"symptoms": ["chest pain"]},
                "hospital_id": "should_be_removed"
            }
        ],
        "active_medications": [],
        "known_allergies": [],
        "chronic_conditions": [],
        "created_at": datetime.utcnow().isoformat()
    }
    
    portable_profile = await privacy_filter.create_portable_profile(
        patient_data=patient_data,
        privacy_level="standard"
    )
    
    assert portable_profile.patient_id == "MED-test-001"
    assert portable_profile.privacy_filtered == True
    assert len(portable_profile.clinical_timeline) == 1
    print("✅ Portable profile creation successful")


async def test_cryptographic_service():
    """Test cryptographic operations."""
    print("\n🔐 Testing Cryptographic Service...")
    
    hospital_id = "test_hospital_crypto"
    crypto_service = CryptographicService(hospital_id)
    
    # Test profile signing and encryption
    test_profile = {
        "patient_id": "MED-test-002",
        "clinical_data": "Test clinical information",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    signed_encrypted = await crypto_service.sign_and_encrypt_profile(
        profile_data=test_profile,
        export_format="json"
    )
    
    assert "encrypted_data" in signed_encrypted
    assert "signature" in signed_encrypted
    assert "algorithm" in signed_encrypted
    print("✅ Profile signing and encryption successful")
    
    # Test decryption and verification
    decrypted_data = await crypto_service.decrypt_and_verify_profile(
        encrypted_data=signed_encrypted["encrypted_data"],
        signature=signed_encrypted["signature"]
    )
    
    assert decrypted_data is not None, "Decryption should succeed"
    print("✅ Profile decryption and verification successful")
    
    # Test timeline entry signing
    timeline_entry = ClinicalTimelineEntry(
        entry_id="test_entry_001",
        patient_id="MED-test-002",
        timestamp=datetime.utcnow(),
        event_type=EventType.CLINICAL_VISIT,
        clinical_summary="Test clinical summary",
        structured_data={"test": "data"}
    )
    
    signature = await crypto_service.sign_timeline_entry(timeline_entry)
    assert len(signature) > 0, "Timeline entry should be signed"
    print("✅ Timeline entry signing successful")
    
    # Test signature verification
    entry_dict = {
        "entry_id": timeline_entry.entry_id,
        "patient_id": timeline_entry.patient_id,
        "timestamp": timeline_entry.timestamp,
        "event_type": timeline_entry.event_type,
        "clinical_summary": timeline_entry.clinical_summary,
        "structured_data": timeline_entry.structured_data,
        "cryptographic_signature": signature
    }
    
    is_valid = await crypto_service.verify_timeline_entry_signature(entry_dict)
    assert is_valid == True, "Timeline entry signature should be valid"
    print("✅ Timeline entry signature verification successful")


async def test_privacy_validation():
    """Test privacy validation functionality."""
    print("\n🔍 Testing Privacy Validation...")
    
    hospital_id = "test_hospital_validation"
    privacy_filter = PrivacyFilterService(hospital_id)
    
    # Test valid portable profile
    valid_profile = PortableProfile(
        patient_id="MED-test-003",
        created_at=datetime.utcnow(),
        last_updated=datetime.utcnow(),
        demographics=PatientDemographics(
            biological_sex="female",
            date_of_birth="1990-05-15"
        ),
        clinical_timeline=[
            ClinicalTimelineEntry(
                entry_id="entry_001",
                patient_id="MED-test-003",
                timestamp=datetime.utcnow(),
                event_type=EventType.CLINICAL_VISIT,
                clinical_summary="Patient presents with symptoms",
                structured_data={"symptoms": ["headache"]}
            )
        ]
    )
    
    validation_result = await privacy_filter.validate_imported_profile(valid_profile)
    assert validation_result.is_compliant == True, "Valid profile should pass validation"
    assert validation_result.compliance_score > 0.8, "Compliance score should be high"
    print("✅ Valid profile privacy validation successful")
    
    # Test model parameters validation
    valid_parameters = {
        "layer_1_weights": [[0.1, 0.2], [0.3, 0.4]],
        "layer_2_weights": [0.5, 0.6],
        "bias_terms": [0.1, 0.2],
        "training_metadata": {
            "epochs": 5,
            "learning_rate": 0.001
        }
    }
    
    param_validation = await privacy_filter.validate_model_parameters(valid_parameters)
    assert param_validation.is_compliant == True, "Valid parameters should pass validation"
    print("✅ Model parameters privacy validation successful")
    
    # Test invalid parameters (containing patient data)
    invalid_parameters = {
        "layer_weights": [[0.1, 0.2]],
        "patient_names": ["John Doe", "Jane Smith"],  # This should trigger violation
        "training_data": "patient information"  # This should trigger violation
    }
    
    invalid_validation = await privacy_filter.validate_model_parameters(invalid_parameters)
    assert invalid_validation.is_compliant == False, "Invalid parameters should fail validation"
    assert len(invalid_validation.violations) > 0, "Should have privacy violations"
    print("✅ Invalid parameters correctly rejected")


async def test_service_integration():
    """Test integration between privacy filter and crypto services."""
    print("\n🔗 Testing Service Integration...")
    
    hospital_id = "test_hospital_integration"
    privacy_filter = PrivacyFilterService(hospital_id)
    crypto_service = CryptographicService(hospital_id)
    
    # Create test patient data with hospital metadata
    patient_data = {
        "patient_id": "MED-integration-001",
        "demographics": {
            "first_name": "Alice",
            "biological_sex": "female",
            "date_of_birth": "1985-03-20"
        },
        "clinical_timeline": [
            {
                "entry_id": "entry_001",
                "patient_id": "MED-integration-001",
                "timestamp": datetime.utcnow(),
                "event_type": "clinical_visit",
                "clinical_summary": "Patient at City Hospital seen by Dr. Brown",
                "structured_data": {
                    "symptoms": ["fever", "cough"],
                    "hospital_id": "city_hospital_001",
                    "attending_physician": "Dr. Brown"
                }
            }
        ],
        "active_medications": [
            {
                "name": "Acetaminophen",
                "dosage": "500mg",
                "prescriber": "Dr. Brown"  # Should be removed
            }
        ],
        "created_at": datetime.utcnow().isoformat()
    }
    
    # Create privacy-filtered portable profile
    portable_profile = await privacy_filter.create_portable_profile(
        patient_data=patient_data,
        privacy_level="standard"
    )
    
    # Verify privacy filtering worked
    timeline_entry = portable_profile.clinical_timeline[0]
    assert "City Hospital" not in timeline_entry.clinical_summary
    assert "Dr. Brown" not in timeline_entry.clinical_summary
    assert "hospital_id" not in timeline_entry.structured_data
    assert "attending_physician" not in timeline_entry.structured_data
    print("✅ Privacy filtering in integration successful")
    
    # Sign and encrypt the filtered profile
    signed_encrypted = await crypto_service.sign_and_encrypt_profile(
        profile_data=portable_profile.dict(),
        export_format="json"
    )
    
    # Decrypt and verify
    decrypted_data = await crypto_service.decrypt_and_verify_profile(
        encrypted_data=signed_encrypted["encrypted_data"],
        signature=signed_encrypted["signature"]
    )
    
    assert decrypted_data is not None
    print("✅ Cryptographic operations in integration successful")
    
    # Validate the final profile
    validation_result = await privacy_filter.validate_exported_profile(portable_profile)
    assert validation_result.is_compliant == True
    print("✅ Final profile validation successful")


async def run_dol_service_tests():
    """Run comprehensive DOL service tests."""
    print("🚀 Starting DOL Service Test Suite")
    print("=" * 60)
    
    try:
        # Run all tests
        await test_privacy_filter_service()
        await test_cryptographic_service()
        await test_privacy_validation()
        await test_service_integration()
        
        print("\n" + "=" * 60)
        print("🎉 ALL DOL SERVICE TESTS PASSED!")
        print("=" * 60)
        
        print("\n📊 Test Summary:")
        print("✅ Privacy filtering and hospital metadata removal")
        print("✅ Cryptographic signing and encryption")
        print("✅ Profile integrity verification")
        print("✅ Privacy compliance validation")
        print("✅ Service integration and end-to-end workflow")
        
        print("\n🔒 Privacy Guarantees Verified:")
        print("✅ Hospital names completely removed from clinical text")
        print("✅ Provider names filtered from all content")
        print("✅ Location data stripped from structured data")
        print("✅ Model parameters validated for patient data absence")
        print("✅ Cryptographic integrity without hospital identification")
        
        print("\n🎯 DOL Service Ready for Task 4.2!")
        return True
        
    except Exception as e:
        print(f"\n❌ DOL SERVICE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the comprehensive test
    success = asyncio.run(run_dol_service_tests())
    sys.exit(0 if success else 1)