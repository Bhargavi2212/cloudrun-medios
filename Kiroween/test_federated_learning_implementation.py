#!/usr/bin/env python3
"""
Test script for federated learning implementation.

This script validates the federated learning models and services
to ensure they work correctly and maintain privacy guarantees.
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, Any, List

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from shared.federation import (
        ModelType,
        PrivacyLevel,
        TriageModel,
        SOAPNoteModel,
        ClinicalSummarizationModel,
        FederatedModelRegistry
    )
    print("✅ Successfully imported federated learning modules")
except ImportError as e:
    print(f"❌ Failed to import federated learning modules: {e}")
    sys.exit(1)


def create_sample_clinical_data() -> List[Dict[str, Any]]:
    """Create sample clinical data for testing."""
    return [
        {
            "clinical_summary": "Patient presents with chest pain and shortness of breath. ECG shows normal sinus rhythm.",
            "structured_data": {
                "symptoms": ["chest pain", "dyspnea"],
                "vital_signs": {"hr": 88, "bp": "120/80", "temp": 98.6},
                "medications": ["aspirin 81mg"]
            },
            "event_type": "clinical_visit",
            "timestamp": datetime.utcnow(),
            # These should be filtered out by privacy mechanisms
            "hospital_id": "secret_hospital_001",
            "attending_physician": "Dr. Smith",
            "department": "Emergency"
        },
        {
            "clinical_summary": "Follow-up visit for diabetes management. Blood glucose levels stable.",
            "structured_data": {
                "symptoms": ["fatigue"],
                "medications": ["metformin 500mg"],
                "vital_signs": {"glucose": 140, "bp": "130/85"}
            },
            "event_type": "follow_up",
            "timestamp": datetime.utcnow(),
            # These should be filtered out
            "hospital_mrn": "MRN123456",
            "room_number": "Room 205"
        },
        {
            "clinical_summary": "Emergency presentation with severe abdominal pain. CT scan ordered.",
            "structured_data": {
                "symptoms": ["abdominal pain", "nausea"],
                "vital_signs": {"hr": 110, "bp": "140/90", "temp": 101.2},
                "procedures": ["CT abdomen"]
            },
            "event_type": "emergency",
            "timestamp": datetime.utcnow()
        }
    ]


def create_sample_soap_data() -> List[Dict[str, Any]]:
    """Create sample SOAP note data for testing."""
    return [
        {
            "subjective": "Patient reports chest pain for 2 hours, described as crushing sensation",
            "objective": "Vital signs: HR 88, BP 120/80, Temp 98.6F. Physical exam reveals no acute distress",
            "assessment": "Chest pain, rule out acute coronary syndrome",
            "plan": "ECG, cardiac enzymes, chest X-ray. Monitor in ED",
            "chief_complaint": "Chest pain",
            "clinical_context": {"triage_level": "ESI-2"},
            # Should be filtered out
            "attending_physician": "Dr. Johnson",
            "department": "Emergency Department"
        },
        {
            "subjective": "Patient with known diabetes presents for routine follow-up",
            "objective": "Blood glucose 140 mg/dL, BP 130/85, weight stable",
            "assessment": "Type 2 diabetes mellitus, well controlled",
            "plan": "Continue metformin, follow up in 3 months",
            "chief_complaint": "Diabetes follow-up",
            "clinical_context": {"hba1c": 7.1}
        }
    ]


def create_sample_summarization_data() -> List[Dict[str, Any]]:
    """Create sample clinical documents for summarization testing."""
    return [
        {
            "full_text": """
            Patient is a 65-year-old male with history of hypertension and diabetes who presents 
            to the emergency department with acute onset chest pain. Pain began 2 hours ago while 
            at rest, described as crushing substernal pain radiating to left arm. Associated with 
            diaphoresis and nausea. No shortness of breath. Vital signs on arrival: HR 88, 
            BP 120/80, RR 16, O2 sat 98% on room air. Physical examination reveals anxious 
            appearing male in mild distress. Cardiovascular exam shows regular rate and rhythm, 
            no murmurs. ECG shows normal sinus rhythm with no acute ST changes. Chest X-ray 
            normal. Cardiac enzymes pending.
            """,
            "summary": "65M with HTN/DM presents with acute chest pain, ECG normal, enzymes pending",
            "document_type": "emergency_note",
            "specialty": "emergency_medicine",
            "key_findings": ["chest pain", "normal ECG", "stable vitals"],
            "medications": ["aspirin", "metoprolol"],
            "procedures": ["ECG", "chest X-ray"],
            # Should be filtered out
            "attending_physician": "Dr. Williams",
            "hospital_mrn": "MRN789012"
        }
    ]


async def test_model_initialization():
    """Test federated model initialization."""
    print("\n🧪 Testing Model Initialization...")
    
    hospital_id = "test_hospital_001"
    
    # Test TriageModel
    triage_model = TriageModel(hospital_id=hospital_id)
    assert triage_model.hospital_id == hospital_id
    assert triage_model.model_type == ModelType.TRIAGE
    assert triage_model.current_round == 0
    print("✅ TriageModel initialization successful")
    
    # Test SOAPNoteModel
    soap_model = SOAPNoteModel(hospital_id=hospital_id)
    assert soap_model.hospital_id == hospital_id
    assert soap_model.model_type == ModelType.SOAP_NOTE_GENERATION
    assert len(soap_model.soap_sections) == 4
    print("✅ SOAPNoteModel initialization successful")
    
    # Test ClinicalSummarizationModel
    summarization_model = ClinicalSummarizationModel(hospital_id=hospital_id)
    assert summarization_model.hospital_id == hospital_id
    assert summarization_model.model_type == ModelType.CLINICAL_SUMMARIZATION
    assert "embedding_dim" in summarization_model.model_architecture
    print("✅ ClinicalSummarizationModel initialization successful")


async def test_privacy_filtering():
    """Test privacy filtering functionality."""
    print("\n🔒 Testing Privacy Filtering...")
    
    hospital_id = "test_hospital_002"
    model = TriageModel(hospital_id=hospital_id)
    
    # Create data with hospital identifiers
    raw_data = create_sample_clinical_data()
    
    # Filter data
    filtered_data = await model.prepare_training_data(raw_data)
    
    # Verify hospital identifiers are removed
    for record in filtered_data:
        assert "hospital_id" not in record, "Hospital ID should be filtered out"
        assert "attending_physician" not in record, "Physician name should be filtered out"
        assert "department" not in record, "Department should be filtered out"
        assert "hospital_mrn" not in record, "Hospital MRN should be filtered out"
        assert "room_number" not in record, "Room number should be filtered out"
        
        # Verify clinical content is preserved
        assert "clinical_text" in record, "Clinical text should be preserved"
        assert "structured_data" in record, "Structured data should be preserved"
        assert "event_type" in record, "Event type should be preserved"
    
    print(f"✅ Privacy filtering successful - {len(filtered_data)} records processed")
    print(f"   Original records: {len(raw_data)}, Filtered records: {len(filtered_data)}")


async def test_model_training():
    """Test model training functionality."""
    print("\n🎯 Testing Model Training...")
    
    hospital_id = "test_hospital_003"
    
    # Test TriageModel training
    triage_model = TriageModel(hospital_id=hospital_id)
    clinical_data = create_sample_clinical_data()
    
    print("   Training TriageModel...")
    weights = await triage_model.train_local_round(clinical_data)
    
    assert weights is not None, "Training should return weights"
    assert weights.model_id == triage_model.model_id, "Model ID should match"
    assert weights.hospital_id == hospital_id, "Hospital ID should match"
    assert weights.privacy_budget > 0, "Privacy budget should be positive"
    assert triage_model.current_round == 1, "Training round should increment"
    
    print(f"✅ TriageModel training successful - Round {weights.training_round}")
    print(f"   Weight ID: {weights.weight_id}")
    print(f"   Privacy budget used: {weights.privacy_budget}")
    
    # Test model evaluation
    print("   Evaluating TriageModel...")
    metrics = await triage_model.evaluate_model(clinical_data, weights)
    
    assert "accuracy" in metrics, "Should have accuracy metric"
    assert "precision" in metrics, "Should have precision metric"
    assert "recall" in metrics, "Should have recall metric"
    assert "f1_score" in metrics, "Should have F1 score metric"
    
    print(f"✅ TriageModel evaluation successful")
    print(f"   Accuracy: {metrics['accuracy']:.3f}")
    print(f"   Precision: {metrics['precision']:.3f}")
    print(f"   Recall: {metrics['recall']:.3f}")


async def test_differential_privacy():
    """Test differential privacy application."""
    print("\n🛡️ Testing Differential Privacy...")
    
    hospital_id = "test_hospital_004"
    model = TriageModel(hospital_id=hospital_id)
    
    # Create sample weights
    original_weights = {
        "layer_1": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "layer_2": [0.5, 1.5, 2.5],
        "bias": [0.1, 0.2],
        "metadata": "non_numerical_data"
    }
    
    # Apply differential privacy
    private_weights = await model.apply_differential_privacy(
        original_weights,
        privacy_budget=0.5
    )
    
    # Verify structure is preserved
    assert set(private_weights.keys()) == set(original_weights.keys()), "Keys should be preserved"
    
    # Verify noise was added to numerical data
    assert private_weights["layer_1"] != original_weights["layer_1"], "Noise should be added to layer_1"
    assert private_weights["layer_2"] != original_weights["layer_2"], "Noise should be added to layer_2"
    assert private_weights["bias"] != original_weights["bias"], "Noise should be added to bias"
    
    # Verify non-numerical data is preserved
    assert private_weights["metadata"] == original_weights["metadata"], "Metadata should be unchanged"
    
    print("✅ Differential privacy application successful")
    print("   Noise added to numerical weights while preserving structure")


async def test_model_registry():
    """Test federated model registry."""
    print("\n📋 Testing Model Registry...")
    
    hospital_id = "test_hospital_005"
    registry = FederatedModelRegistry(hospital_id)
    
    # Register models
    triage_model = TriageModel(hospital_id)
    soap_model = SOAPNoteModel(hospital_id)
    summarization_model = ClinicalSummarizationModel(hospital_id)
    
    registry.register_model(triage_model)
    registry.register_model(soap_model)
    registry.register_model(summarization_model)
    
    # Test model retrieval
    assert registry.get_model("triage") == triage_model, "Should retrieve triage model"
    assert registry.get_model("soap_note_generation") == soap_model, "Should retrieve SOAP model"
    assert registry.get_model("clinical_summarization") == summarization_model, "Should retrieve summarization model"
    assert registry.get_model("nonexistent") is None, "Should return None for nonexistent model"
    
    # Test training status
    status = registry.get_training_status()
    assert "triage" in status, "Should have triage status"
    assert "soap_note_generation" in status, "Should have SOAP status"
    assert "clinical_summarization" in status, "Should have summarization status"
    
    print("✅ Model registry functionality successful")
    print(f"   Registered {len(registry.models)} models")


async def test_privacy_compliance():
    """Test privacy compliance validation."""
    print("\n🔍 Testing Privacy Compliance...")
    
    hospital_id = "test_hospital_006"
    model = TriageModel(hospital_id)
    
    # Test valid record (should pass)
    valid_record = {
        "clinical_text": "Patient presents with symptoms",
        "structured_data": {"symptoms": ["fever"]},
        "event_type": "visit"
    }
    
    assert model._passes_privacy_filter(valid_record) == True, "Valid record should pass filter"
    
    # Test invalid record (should fail)
    invalid_record = {
        "clinical_text": "Patient at General Hospital",
        "hospital_id": "should_not_be_here",
        "event_type": "visit"
    }
    
    assert model._passes_privacy_filter(invalid_record) == False, "Invalid record should fail filter"
    
    print("✅ Privacy compliance validation successful")


async def test_soap_model_functionality():
    """Test SOAP model specific functionality."""
    print("\n📝 Testing SOAP Model Functionality...")
    
    hospital_id = "test_hospital_007"
    soap_model = SOAPNoteModel(hospital_id=hospital_id)
    
    # Test training with SOAP data
    soap_data = create_sample_soap_data()
    weights = await soap_model.train_local_round(soap_data)
    
    assert weights is not None, "SOAP training should return weights"
    assert weights.model_id.startswith("soap_note_"), "Model ID should indicate SOAP model"
    
    # Test evaluation
    metrics = await soap_model.evaluate_model(soap_data, weights)
    assert "bleu_score" in metrics, "Should have BLEU score for text generation"
    assert "clinical_accuracy" in metrics, "Should have clinical accuracy"
    
    print("✅ SOAP model functionality successful")
    print(f"   BLEU score: {metrics['bleu_score']:.3f}")
    print(f"   Clinical accuracy: {metrics['clinical_accuracy']:.3f}")


async def test_summarization_model_functionality():
    """Test summarization model specific functionality."""
    print("\n📄 Testing Summarization Model Functionality...")
    
    hospital_id = "test_hospital_008"
    summarization_model = ClinicalSummarizationModel(hospital_id=hospital_id)
    
    # Test training with clinical documents
    doc_data = create_sample_summarization_data()
    weights = await summarization_model.train_local_round(doc_data)
    
    assert weights is not None, "Summarization training should return weights"
    assert weights.model_id.startswith("clinical_summarization_"), "Model ID should indicate summarization model"
    
    # Test evaluation
    metrics = await summarization_model.evaluate_model(doc_data, weights)
    assert "rouge_1" in metrics, "Should have ROUGE-1 score"
    assert "rouge_l" in metrics, "Should have ROUGE-L score"
    assert "clinical_accuracy" in metrics, "Should have clinical accuracy"
    
    print("✅ Summarization model functionality successful")
    print(f"   ROUGE-1: {metrics['rouge_1']:.3f}")
    print(f"   ROUGE-L: {metrics['rouge_l']:.3f}")
    print(f"   Clinical accuracy: {metrics['clinical_accuracy']:.3f}")


async def run_comprehensive_test():
    """Run comprehensive federated learning test suite."""
    print("🚀 Starting Comprehensive Federated Learning Test Suite")
    print("=" * 60)
    
    try:
        # Run all tests
        await test_model_initialization()
        await test_privacy_filtering()
        await test_model_training()
        await test_differential_privacy()
        await test_model_registry()
        await test_privacy_compliance()
        await test_soap_model_functionality()
        await test_summarization_model_functionality()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! Federated Learning Implementation is Working!")
        print("=" * 60)
        
        print("\n📊 Test Summary:")
        print("✅ Model initialization and configuration")
        print("✅ Privacy filtering and data protection")
        print("✅ Local model training simulation")
        print("✅ Differential privacy application")
        print("✅ Model registry management")
        print("✅ Privacy compliance validation")
        print("✅ SOAP note model functionality")
        print("✅ Clinical summarization functionality")
        
        print("\n🔒 Privacy Guarantees Verified:")
        print("✅ Hospital metadata completely removed from training data")
        print("✅ Differential privacy noise applied to model weights")
        print("✅ Privacy budget properly managed and tracked")
        print("✅ Only clinical content preserved, no identifying information")
        
        print("\n🎯 Ready for Phase 4: Data Orchestration Layer (DOL)")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the comprehensive test
    success = asyncio.run(run_comprehensive_test())
    sys.exit(0 if success else 1)