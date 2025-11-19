"""
Tests for federated learning functionality.

This module tests the federated learning models and services
to ensure privacy preservation and functionality.
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any, List

from shared.federation import (
    ModelType,
    PrivacyLevel,
    TriageModel,
    SOAPNoteModel,
    ClinicalSummarizationModel,
    FederatedModelRegistry
)


class TestFederatedModels:
    """Test federated learning models."""
    
    @pytest.fixture
    def sample_clinical_data(self) -> List[Dict[str, Any]]:
        """Sample clinical data for testing."""
        return [
            {
                "clinical_summary": "Patient presents with chest pain and shortness of breath",
                "structured_data": {
                    "symptoms": ["chest pain", "dyspnea"],
                    "vital_signs": {"hr": 88, "bp": "120/80"}
                },
                "event_type": "clinical_visit",
                "timestamp": datetime.utcnow()
            },
            {
                "clinical_summary": "Follow-up visit for diabetes management",
                "structured_data": {
                    "symptoms": ["fatigue"],
                    "medications": ["metformin"],
                    "vital_signs": {"glucose": 140}
                },
                "event_type": "follow_up",
                "timestamp": datetime.utcnow()
            }
        ]
    
    @pytest.mark.asyncio
    async def test_triage_model_initialization(self):
        """Test triage model initialization."""
        hospital_id = "test_hospital_001"
        model = TriageModel(hospital_id=hospital_id)
        
        assert model.hospital_id == hospital_id
        assert model.model_type == ModelType.TRIAGE
        assert model.privacy_level == PrivacyLevel.STANDARD
        assert model.current_round == 0
        assert len(model.training_history) == 0
    
    @pytest.mark.asyncio
    async def test_soap_model_initialization(self):
        """Test SOAP note model initialization."""
        hospital_id = "test_hospital_002"
        model = SOAPNoteModel(hospital_id=hospital_id)
        
        assert model.hospital_id == hospital_id
        assert model.model_type == ModelType.SOAP_NOTE_GENERATION
        assert model.privacy_level == PrivacyLevel.STANDARD
        assert len(model.soap_sections) == 4
    
    @pytest.mark.asyncio
    async def test_summarization_model_initialization(self):
        """Test clinical summarization model initialization."""
        hospital_id = "test_hospital_003"
        model = ClinicalSummarizationModel(hospital_id=hospital_id)
        
        assert model.hospital_id == hospital_id
        assert model.model_type == ModelType.CLINICAL_SUMMARIZATION
        assert model.privacy_level == PrivacyLevel.STANDARD
        assert "embedding_dim" in model.model_architecture
    
    @pytest.mark.asyncio
    async def test_privacy_data_filtering(self, sample_clinical_data):
        """Test that privacy filtering removes hospital identifiers."""
        hospital_id = "test_hospital_004"
        model = TriageModel(hospital_id=hospital_id)
        
        # Add hospital-identifying information to test data
        test_data = sample_clinical_data.copy()
        test_data[0]["hospital_id"] = "should_be_removed"
        test_data[0]["attending_physician"] = "Dr. Smith"
        
        filtered_data = await model.prepare_training_data(test_data)
        
        # Verify hospital identifiers are removed
        for record in filtered_data:
            assert "hospital_id" not in record
            assert "attending_physician" not in record
            assert "clinical_text" in record
            assert "structured_data" in record
    
    @pytest.mark.asyncio
    async def test_privacy_filter_validation(self, sample_clinical_data):
        """Test privacy filter validation."""
        hospital_id = "test_hospital_005"
        model = TriageModel(hospital_id=hospital_id)
        
        # Test record that should pass privacy filter
        valid_record = {
            "clinical_text": "Patient presents with symptoms",
            "structured_data": {"symptoms": ["fever"]},
            "event_type": "visit"
        }
        
        assert model._passes_privacy_filter(valid_record) == True
        
        # Test record that should fail privacy filter
        invalid_record = {
            "clinical_text": "Patient at General Hospital",
            "hospital_id": "should_not_be_here",
            "event_type": "visit"
        }
        
        assert model._passes_privacy_filter(invalid_record) == False
    
    @pytest.mark.asyncio
    async def test_model_training_simulation(self, sample_clinical_data):
        """Test simulated model training."""
        hospital_id = "test_hospital_006"
        model = TriageModel(hospital_id=hospital_id)
        
        # Train one round
        weights = await model.train_local_round(sample_clinical_data)
        
        assert weights is not None
        assert weights.model_id == model.model_id
        assert weights.hospital_id == hospital_id
        assert weights.training_round == 0
        assert weights.privacy_budget > 0
        assert model.current_round == 1
        assert len(model.training_history) == 1
    
    @pytest.mark.asyncio
    async def test_model_evaluation(self, sample_clinical_data):
        """Test model evaluation."""
        hospital_id = "test_hospital_007"
        model = TriageModel(hospital_id=hospital_id)
        
        # Train model first
        weights = await model.train_local_round(sample_clinical_data)
        
        # Evaluate model
        metrics = await model.evaluate_model(sample_clinical_data, weights)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert metrics["samples_evaluated"] == len(sample_clinical_data)
    
    @pytest.mark.asyncio
    async def test_differential_privacy_application(self, sample_clinical_data):
        """Test differential privacy application."""
        hospital_id = "test_hospital_008"
        model = TriageModel(hospital_id=hospital_id)
        
        # Create sample weights
        original_weights = {
            "layer_1": [[1.0, 2.0], [3.0, 4.0]],
            "layer_2": [0.5, 1.5],
            "metadata": "non_numerical_data"
        }
        
        # Apply differential privacy
        private_weights = await model.apply_differential_privacy(
            original_weights,
            privacy_budget=0.5
        )
        
        # Verify structure is preserved
        assert "layer_1" in private_weights
        assert "layer_2" in private_weights
        assert "metadata" in private_weights
        
        # Verify noise was added to numerical data
        assert private_weights["layer_1"] != original_weights["layer_1"]
        assert private_weights["layer_2"] != original_weights["layer_2"]
        
        # Verify non-numerical data is preserved
        assert private_weights["metadata"] == original_weights["metadata"]


class TestFederatedModelRegistry:
    """Test federated model registry."""
    
    def test_model_registry_initialization(self):
        """Test model registry initialization."""
        hospital_id = "test_hospital_registry"
        registry = FederatedModelRegistry(hospital_id)
        
        assert registry.hospital_id == hospital_id
        assert len(registry.models) == 0
        assert len(registry.global_weights_cache) == 0
    
    def test_model_registration(self):
        """Test model registration."""
        hospital_id = "test_hospital_registry"
        registry = FederatedModelRegistry(hospital_id)
        
        # Register a triage model
        triage_model = TriageModel(hospital_id)
        registry.register_model(triage_model)
        
        assert "triage" in registry.models
        assert registry.get_model("triage") == triage_model
        assert registry.get_model("nonexistent") is None
    
    def test_training_status(self):
        """Test training status retrieval."""
        hospital_id = "test_hospital_registry"
        registry = FederatedModelRegistry(hospital_id)
        
        # Register models
        triage_model = TriageModel(hospital_id)
        soap_model = SOAPNoteModel(hospital_id)
        registry.register_model(triage_model)
        registry.register_model(soap_model)
        
        status = registry.get_training_status()
        
        assert "triage" in status
        assert "soap_note_generation" in status
        assert "current_round" in status["triage"]
        assert "privacy_budget_remaining" in status["triage"]


class TestPrivacyCompliance:
    """Test privacy compliance features."""
    
    @pytest.mark.asyncio
    async def test_no_hospital_metadata_in_training_data(self):
        """Test that no hospital metadata appears in training data."""
        hospital_id = "test_hospital_privacy"
        model = TriageModel(hospital_id)
        
        # Create data with hospital metadata
        raw_data = [
            {
                "clinical_summary": "Patient visit",
                "hospital_id": "secret_hospital",
                "attending_physician": "Dr. Secret",
                "department": "Emergency",
                "room_number": "101"
            }
        ]
        
        filtered_data = await model.prepare_training_data(raw_data)
        
        # Verify all hospital metadata is removed
        for record in filtered_data:
            assert "hospital_id" not in record
            assert "attending_physician" not in record
            assert "department" not in record
            assert "room_number" not in record
    
    def test_privacy_budget_management(self):
        """Test privacy budget management."""
        hospital_id = "test_hospital_budget"
        model = TriageModel(hospital_id)
        
        # Check initial budget
        initial_budget = model.privacy_accountant.get_remaining_budget()
        assert initial_budget > 0
        
        # Allocate budget
        success = model.privacy_accountant.allocate_budget(1, 0.5)
        assert success == True
        
        # Check remaining budget
        remaining_budget = model.privacy_accountant.get_remaining_budget()
        assert remaining_budget == initial_budget - 0.5
        
        # Try to allocate more than remaining
        success = model.privacy_accountant.allocate_budget(2, remaining_budget + 1.0)
        assert success == False


if __name__ == "__main__":
    # Run basic tests
    asyncio.run(test_basic_functionality())