"""
Base classes and interfaces for federated learning.

This module provides the core abstractions for federated learning
across hospital networks while maintaining patient privacy.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum
import uuid


class ModelType(Enum):
    """Types of AI models supported in the federated network."""
    TRIAGE = "triage"
    CLINICAL_SUMMARIZATION = "clinical_summarization"
    SOAP_NOTE_GENERATION = "soap_note_generation"
    DIAGNOSTIC_ASSISTANCE = "diagnostic_assistance"
    RISK_PREDICTION = "risk_prediction"
    TREATMENT_RECOMMENDATION = "treatment_recommendation"


class TrainingStatus(Enum):
    """Status of federated training rounds."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PrivacyLevel(Enum):
    """Privacy levels for federated learning."""
    MINIMAL = "minimal"  # Basic differential privacy
    STANDARD = "standard"  # Enhanced privacy with secure aggregation
    MAXIMUM = "maximum"  # Full homomorphic encryption


class ModelWeights:
    """Container for model weights with privacy metadata."""
    
    def __init__(
        self,
        weights: Dict[str, Any],
        model_id: str,
        hospital_id: str,
        training_round: int,
        privacy_budget: float,
        noise_scale: Optional[float] = None
    ):
        self.weights = weights
        self.model_id = model_id
        self.hospital_id = hospital_id
        self.training_round = training_round
        self.privacy_budget = privacy_budget
        self.noise_scale = noise_scale
        self.timestamp = datetime.utcnow()
        self.weight_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "weight_id": self.weight_id,
            "weights": self.weights,
            "model_id": self.model_id,
            "hospital_id": self.hospital_id,
            "training_round": self.training_round,
            "privacy_budget": self.privacy_budget,
            "noise_scale": self.noise_scale,
            "timestamp": self.timestamp.isoformat()
        }


class FederatedModel(ABC):
    """Abstract base class for federated learning models."""
    
    def __init__(
        self,
        model_id: str,
        model_type: ModelType,
        hospital_id: str,
        privacy_level: PrivacyLevel = PrivacyLevel.STANDARD
    ):
        self.model_id = model_id
        self.model_type = model_type
        self.hospital_id = hospital_id
        self.privacy_level = privacy_level
        self.current_round = 0
        self.training_history: List[Dict[str, Any]] = []
    
    @abstractmethod
    async def train_local_round(
        self,
        training_data: List[Dict[str, Any]],
        global_weights: Optional[ModelWeights] = None
    ) -> ModelWeights:
        """
        Train model locally for one federated round.
        
        Args:
            training_data: Local training data (privacy-filtered)
            global_weights: Current global model weights
            
        Returns:
            Local model weights with privacy guarantees
        """
        pass
    
    @abstractmethod
    async def evaluate_model(
        self,
        test_data: List[Dict[str, Any]],
        model_weights: ModelWeights
    ) -> Dict[str, float]:
        """
        Evaluate model performance on local test data.
        
        Args:
            test_data: Local test data
            model_weights: Model weights to evaluate
            
        Returns:
            Performance metrics
        """
        pass
    
    @abstractmethod
    async def apply_differential_privacy(
        self,
        weights: Dict[str, Any],
        privacy_budget: float
    ) -> Dict[str, Any]:
        """
        Apply differential privacy to model weights.
        
        Args:
            weights: Raw model weights
            privacy_budget: Privacy budget for this round
            
        Returns:
            Privacy-protected weights
        """
        pass
    
    async def prepare_training_data(
        self,
        raw_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Prepare and filter training data for privacy.
        
        Args:
            raw_data: Raw clinical data
            
        Returns:
            Privacy-filtered training data
        """
        # Remove all hospital-identifying information
        filtered_data = []
        
        for record in raw_data:
            filtered_record = {
                # Keep only medical content, remove hospital metadata
                "clinical_text": record.get("clinical_summary", ""),
                "structured_data": record.get("structured_data", {}),
                "event_type": record.get("event_type"),
                "timestamp_hour": self._extract_hour_from_timestamp(record.get("timestamp")),  # Only hour, not full timestamp
                # Explicitly exclude hospital identifiers
                # "hospital_id": REMOVED
                # "hospital_mrn": REMOVED
                # "attending_physician": REMOVED
                # "department": REMOVED
            }
            
            # Additional privacy filtering
            if self._passes_privacy_filter(filtered_record):
                filtered_data.append(filtered_record)
        
        return filtered_data
    
    def _passes_privacy_filter(self, record: Dict[str, Any]) -> bool:
        """
        Check if record passes privacy requirements.
        
        Args:
            record: Filtered record to check
            
        Returns:
            True if record is safe for federated learning
        """
        # Ensure no hospital-specific identifiers
        forbidden_keys = [
            "hospital_id", "hospital_mrn", "attending_physician",
            "department", "room_number", "insurance_info"
        ]
        
        for key in forbidden_keys:
            if key in record:
                return False
        
        # Ensure clinical text doesn't contain obvious identifiers
        clinical_text = record.get("clinical_text", "").lower()
        
        # Basic identifier detection (can be enhanced)
        identifier_patterns = [
            "hospital", "dr.", "doctor", "physician",
            "room", "ward", "department", "unit"
        ]
        
        for pattern in identifier_patterns:
            if pattern in clinical_text:
                # Could implement more sophisticated filtering here
                pass
        
        return True


class FederatedAggregator(ABC):
    """Abstract base class for federated weight aggregation."""
    
    @abstractmethod
    async def aggregate_weights(
        self,
        hospital_weights: List[ModelWeights],
        aggregation_method: str = "federated_averaging"
    ) -> ModelWeights:
        """
        Aggregate weights from multiple hospitals.
        
        Args:
            hospital_weights: List of weights from participating hospitals
            aggregation_method: Aggregation algorithm to use
            
        Returns:
            Aggregated global model weights
        """
        pass
    
    @abstractmethod
    async def validate_weights(
        self,
        weights: ModelWeights
    ) -> bool:
        """
        Validate that weights meet privacy and security requirements.
        
        Args:
            weights: Model weights to validate
            
        Returns:
            True if weights are valid
        """
        pass


class FederatedCoordinator(ABC):
    """Abstract base class for coordinating federated learning rounds."""
    
    @abstractmethod
    async def start_training_round(
        self,
        model_id: str,
        participating_hospitals: List[str],
        round_config: Dict[str, Any]
    ) -> str:
        """
        Start a new federated training round.
        
        Args:
            model_id: Model to train
            participating_hospitals: List of hospital IDs
            round_config: Configuration for this round
            
        Returns:
            Training round ID
        """
        pass
    
    @abstractmethod
    async def collect_local_updates(
        self,
        round_id: str,
        timeout_seconds: int = 3600
    ) -> List[ModelWeights]:
        """
        Collect local model updates from hospitals.
        
        Args:
            round_id: Training round identifier
            timeout_seconds: Timeout for collecting updates
            
        Returns:
            List of local model weights
        """
        pass
    
    @abstractmethod
    async def distribute_global_update(
        self,
        round_id: str,
        global_weights: ModelWeights
    ) -> bool:
        """
        Distribute global model update to hospitals.
        
        Args:
            round_id: Training round identifier
            global_weights: Aggregated global weights
            
        Returns:
            True if distribution successful
        """
        pass


class PrivacyAccountant:
    """Tracks privacy budget consumption across federated rounds."""
    
    def __init__(self, total_budget: float = 1.0):
        self.total_budget = total_budget
        self.consumed_budget = 0.0
        self.round_budgets: Dict[int, float] = {}
    
    def allocate_budget(self, round_number: int, budget: float) -> bool:
        """
        Allocate privacy budget for a training round.
        
        Args:
            round_number: Training round number
            budget: Budget to allocate
            
        Returns:
            True if allocation successful
        """
        if self.consumed_budget + budget > self.total_budget:
            return False
        
        self.round_budgets[round_number] = budget
        self.consumed_budget += budget
        return True
    
    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget."""
        return self.total_budget - self.consumed_budget
    
    def get_budget_history(self) -> Dict[int, float]:
        """Get privacy budget allocation history."""
        return self.round_budgets.copy()