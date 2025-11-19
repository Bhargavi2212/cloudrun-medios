"""
Federated learning package for Medi OS Kiroween Edition v2.0.

This package provides privacy-preserving federated learning capabilities
for medical AI models across hospital networks.
"""

from .base import (
    ModelType,
    TrainingStatus,
    PrivacyLevel,
    ModelWeights,
    FederatedModel,
    FederatedAggregator,
    FederatedCoordinator,
    PrivacyAccountant
)

from .models import (
    TriageModel,
    SOAPNoteModel,
    ClinicalSummarizationModel,
    FederatedModelRegistry
)

from .aggregation import (
    FedAvgAggregator,
    SecureAggregator
)

from .transport import (
    SecureTransport,
    FederatedTransportManager,
    PrivacyPreservingTransport
)

__all__ = [
    # Base classes and enums
    "ModelType",
    "TrainingStatus", 
    "PrivacyLevel",
    "ModelWeights",
    "FederatedModel",
    "FederatedAggregator",
    "FederatedCoordinator",
    "PrivacyAccountant",
    
    # Concrete model implementations
    "TriageModel",
    "SOAPNoteModel", 
    "ClinicalSummarizationModel",
    "FederatedModelRegistry",
    
    # Aggregation algorithms
    "FedAvgAggregator",
    "SecureAggregator",
    
    # Secure transport
    "SecureTransport",
    "FederatedTransportManager",
    "PrivacyPreservingTransport"
]

# Package metadata
__version__ = "2.0.0"
__author__ = "Medi OS Kiroween Team"
__description__ = "Privacy-preserving federated learning for medical AI"