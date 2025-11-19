"""
Federated learning aggregation algorithms.

This module implements concrete aggregation algorithms for combining
model weights from multiple hospitals while preserving privacy.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from .base import (
    FederatedAggregator,
    ModelWeights,
    ModelType,
    PrivacyLevel
)


class FedAvgAggregator(FederatedAggregator):
    """
    Federated Averaging (FedAvg) aggregation algorithm.
    
    Implements the standard FedAvg algorithm with privacy validation
    and secure weight combination for medical AI models.
    """
    
    def __init__(self, privacy_level: PrivacyLevel = PrivacyLevel.STANDARD):
        self.privacy_level = privacy_level
        self.aggregation_history: List[Dict[str, Any]] = []
    
    async def aggregate_weights(
        self,
        hospital_weights: List[ModelWeights],
        aggregation_method: str = "federated_averaging"
    ) -> ModelWeights:
        """
        Aggregate weights using Federated Averaging algorithm.
        
        Args:
            hospital_weights: List of weights from participating hospitals
            aggregation_method: Aggregation algorithm ("federated_averaging")
            
        Returns:
            Aggregated global model weights
        """
        if not hospital_weights:
            raise ValueError("No hospital weights provided for aggregation")
        
        # Validate all weights before aggregation
        for weights in hospital_weights:
            if not await self.validate_weights(weights):
                raise ValueError(f"Invalid weights from hospital {weights.hospital_id}")
        
        # Ensure all weights are from the same model and training round
        model_id = hospital_weights[0].model_id
        training_round = hospital_weights[0].training_round
        
        for weights in hospital_weights:
            if weights.model_id != model_id:
                raise ValueError("All weights must be from the same model")
            if weights.training_round != training_round:
                raise ValueError("All weights must be from the same training round")
        
        if aggregation_method == "federated_averaging":
            aggregated_weights = await self._federated_averaging(hospital_weights)
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")
        
        # Create aggregated ModelWeights object
        global_weights = ModelWeights(
            weights=aggregated_weights,
            model_id=model_id,
            hospital_id="global_aggregator",
            training_round=training_round,
            privacy_budget=self._calculate_aggregated_privacy_budget(hospital_weights),
            noise_scale=self._calculate_aggregated_noise_scale(hospital_weights)
        )
        
        # Record aggregation in history
        self.aggregation_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "model_id": model_id,
            "training_round": training_round,
            "participating_hospitals": len(hospital_weights),
            "hospital_ids": [w.hospital_id for w in hospital_weights],
            "aggregation_method": aggregation_method,
            "global_weight_id": global_weights.weight_id
        })
        
        return global_weights
    
    async def _federated_averaging(
        self,
        hospital_weights: List[ModelWeights]
    ) -> Dict[str, Any]:
        """
        Implement the FedAvg algorithm.
        
        Computes weighted average of model parameters where weights
        are proportional to the number of training samples.
        """
        if not hospital_weights:
            return {}
        
        # For simplicity, assume equal weighting (can be enhanced with sample counts)
        num_hospitals = len(hospital_weights)
        
        # Initialize aggregated weights with zeros
        aggregated = {}
        first_weights = hospital_weights[0].weights
        
        # Initialize structure based on first hospital's weights
        aggregated = self._initialize_aggregated_structure(first_weights)
        
        # Sum all weights
        for hospital_weight in hospital_weights:
            self._add_weights_to_aggregated(aggregated, hospital_weight.weights)
        
        # Average the weights
        self._divide_weights_by_scalar(aggregated, num_hospitals)
        
        return aggregated
    
    def _initialize_aggregated_structure(self, weights: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize aggregated weights structure with zeros."""
        def init_zeros(obj):
            if isinstance(obj, dict):
                return {k: init_zeros(v) for k, v in obj.items()}
            elif isinstance(obj, (list, np.ndarray)):
                return np.zeros_like(np.array(obj)).tolist()
            else:
                return obj
        
        return init_zeros(weights)
    
    def _add_weights_to_aggregated(
        self,
        aggregated: Dict[str, Any],
        weights: Dict[str, Any]
    ) -> None:
        """Add weights to aggregated structure."""
        def add_recursive(agg_obj, weight_obj):
            if isinstance(agg_obj, dict) and isinstance(weight_obj, dict):
                for key in agg_obj:
                    if key in weight_obj:
                        add_recursive(agg_obj[key], weight_obj[key])
            elif isinstance(agg_obj, list) and isinstance(weight_obj, list):
                agg_array = np.array(agg_obj)
                weight_array = np.array(weight_obj)
                result = agg_array + weight_array
                # Update the list in place
                for i, val in enumerate(result.tolist()):
                    agg_obj[i] = val
        
        add_recursive(aggregated, weights)
    
    def _divide_weights_by_scalar(
        self,
        weights: Dict[str, Any],
        divisor: float
    ) -> None:
        """Divide all weights by a scalar value."""
        def divide_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    divide_recursive(value)
            elif isinstance(obj, list):
                array = np.array(obj)
                result = array / divisor
                # Update the list in place
                for i, val in enumerate(result.tolist()):
                    obj[i] = val
        
        divide_recursive(weights)
    
    async def validate_weights(self, weights: ModelWeights) -> bool:
        """
        Validate that weights meet privacy and security requirements.
        
        Args:
            weights: Model weights to validate
            
        Returns:
            True if weights are valid
        """
        # Check basic structure
        if not weights.weights or not isinstance(weights.weights, dict):
            return False
        
        # Validate privacy budget
        if weights.privacy_budget <= 0:
            return False
        
        # Check for potential privacy violations
        if not self._check_privacy_compliance(weights):
            return False
        
        # Validate weight structure integrity
        if not self._validate_weight_structure(weights.weights):
            return False
        
        return True
    
    def _check_privacy_compliance(self, weights: ModelWeights) -> bool:
        """
        Check if weights comply with privacy requirements.
        
        Ensures no patient data or hospital identifiers are embedded.
        """
        # Check that hospital_id is properly anonymized for global aggregation
        if weights.hospital_id == "global_aggregator":
            return True  # Global weights are allowed
        
        # Ensure privacy budget is within acceptable range
        if weights.privacy_budget > 2.0:  # Conservative threshold
            return False
        
        # Check for suspicious patterns in weights that might indicate data leakage
        if not self._detect_potential_data_leakage(weights.weights):
            return False
        
        return True
    
    def _detect_potential_data_leakage(self, weights: Dict[str, Any]) -> bool:
        """
        Detect potential patient data leakage in model weights.
        
        This is a simplified check - in production, more sophisticated
        analysis would be needed.
        """
        def check_recursive(obj, path=""):
            if isinstance(obj, dict):
                # Check for suspicious keys that might contain patient data
                suspicious_keys = [
                    "patient_id", "mrn", "ssn", "name", "address",
                    "phone", "email", "dob", "insurance"
                ]
                
                for key in obj.keys():
                    if any(sus_key in key.lower() for sus_key in suspicious_keys):
                        return False
                    if not check_recursive(obj[key], f"{path}.{key}"):
                        return False
                        
            elif isinstance(obj, (list, np.ndarray)):
                # Check for extremely large values that might indicate embedded data
                try:
                    array = np.array(obj)
                    if np.any(np.abs(array) > 1000):  # Threshold for suspicious values
                        return False
                except (ValueError, TypeError):
                    pass
            
            return True
        
        return check_recursive(weights)
    
    def _validate_weight_structure(self, weights: Dict[str, Any]) -> bool:
        """
        Validate the structure of model weights.
        
        Ensures weights have expected neural network structure.
        """
        # Check for required top-level keys (varies by model type)
        expected_structures = [
            ["input_layer", "hidden_layer_1", "output_layer"],  # Basic NN
            ["embedding_layer", "transformer_layers", "output_projection"],  # Transformer
            ["attention_weights", "output_layer"]  # Attention model
        ]
        
        # Check if weights match any expected structure
        for expected in expected_structures:
            if all(key in weights for key in expected):
                return True
        
        # If no standard structure matches, do basic validation
        if len(weights) == 0:
            return False
        
        # Ensure all values are serializable
        try:
            import json
            json.dumps(weights, default=str)
        except (TypeError, ValueError):
            return False
        
        return True
    
    def _calculate_aggregated_privacy_budget(
        self,
        hospital_weights: List[ModelWeights]
    ) -> float:
        """
        Calculate the privacy budget for aggregated weights.
        
        Uses composition theorem for differential privacy.
        """
        # Simple composition - sum of individual budgets
        # In production, use more sophisticated composition theorems
        total_budget = sum(w.privacy_budget for w in hospital_weights)
        
        # Apply privacy amplification due to aggregation
        amplification_factor = 1.0 / len(hospital_weights)
        
        return total_budget * amplification_factor
    
    def _calculate_aggregated_noise_scale(
        self,
        hospital_weights: List[ModelWeights]
    ) -> Optional[float]:
        """
        Calculate the effective noise scale for aggregated weights.
        """
        noise_scales = [w.noise_scale for w in hospital_weights if w.noise_scale is not None]
        
        if not noise_scales:
            return None
        
        # RMS combination of noise scales
        rms_noise = np.sqrt(np.mean([ns**2 for ns in noise_scales]))
        
        # Noise reduction due to averaging
        reduction_factor = 1.0 / np.sqrt(len(hospital_weights))
        
        return rms_noise * reduction_factor
    
    def get_aggregation_history(self) -> List[Dict[str, Any]]:
        """Get history of aggregation operations."""
        return self.aggregation_history.copy()


class SecureAggregator(FederatedAggregator):
    """
    Secure aggregation with enhanced privacy guarantees.
    
    Implements secure multi-party computation for weight aggregation
    with additional privacy protections.
    """
    
    def __init__(self, privacy_level: PrivacyLevel = PrivacyLevel.MAXIMUM):
        self.privacy_level = privacy_level
        self.aggregation_history: List[Dict[str, Any]] = []
    
    async def aggregate_weights(
        self,
        hospital_weights: List[ModelWeights],
        aggregation_method: str = "secure_aggregation"
    ) -> ModelWeights:
        """
        Aggregate weights using secure multi-party computation.
        
        This is a placeholder for more advanced secure aggregation.
        In production, this would implement protocols like:
        - Secure aggregation with dropout resilience
        - Homomorphic encryption-based aggregation
        - Secret sharing-based protocols
        """
        # For now, use FedAvg with additional privacy checks
        fedavg_aggregator = FedAvgAggregator(self.privacy_level)
        
        # Apply additional privacy validation
        for weights in hospital_weights:
            if not await self._enhanced_privacy_validation(weights):
                raise ValueError(f"Enhanced privacy validation failed for {weights.hospital_id}")
        
        # Use FedAvg as base algorithm
        global_weights = await fedavg_aggregator.aggregate_weights(
            hospital_weights,
            "federated_averaging"
        )
        
        # Apply additional noise for maximum privacy
        if self.privacy_level == PrivacyLevel.MAXIMUM:
            global_weights.weights = await self._apply_additional_noise(global_weights.weights)
        
        return global_weights
    
    async def validate_weights(self, weights: ModelWeights) -> bool:
        """Enhanced validation for secure aggregation."""
        # Use base validation
        fedavg_aggregator = FedAvgAggregator()
        base_valid = await fedavg_aggregator.validate_weights(weights)
        
        if not base_valid:
            return False
        
        # Additional security checks
        return await self._enhanced_privacy_validation(weights)
    
    async def _enhanced_privacy_validation(self, weights: ModelWeights) -> bool:
        """
        Enhanced privacy validation for maximum security.
        """
        # Check for minimum noise scale
        if weights.noise_scale is None or weights.noise_scale < 0.1:
            return False
        
        # Ensure privacy budget is conservative
        if weights.privacy_budget > 0.5:
            return False
        
        # Additional checks for secure aggregation
        # (In production, implement cryptographic verification)
        
        return True
    
    async def _apply_additional_noise(self, weights: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply additional noise for maximum privacy level.
        """
        additional_noise_scale = 0.01
        
        def add_noise_recursive(obj):
            if isinstance(obj, dict):
                return {k: add_noise_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, (list, np.ndarray)):
                array = np.array(obj)
                noise = np.random.normal(0, additional_noise_scale, array.shape)
                return (array + noise).tolist()
            else:
                return obj
        
        return add_noise_recursive(weights)