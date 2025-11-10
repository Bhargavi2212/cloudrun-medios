"""
ESI 5-Class Severity Mapper

Maps 7 ESI classes to 5 severity levels for easier classification.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict


class ESI5ClassMapper(BaseEstimator, TransformerMixin):
    """
    Maps 7 ESI classes to 5 severity levels.
    
    Mapping:
    - ESI 0 (resuscitation) → Severity 1 (Critical)
    - ESI 1 (critical)     → Severity 1 (Critical)
    - ESI 2 (emergent)     → Severity 2 (Emergent)
    - ESI 3 (urgent)       → Severity 3 (Urgent)
    - ESI 4 (less urgent)   → Severity 4 (Standard)
    - ESI 5 (non-urgent)   → Severity 5 (Non-urgent)
    - ESI 7 (unknown)      → Severity 4 (Standard)
    """
    
    # Forward mapping: 7-class ESI → 5-class Severity
    ESI_TO_SEVERITY = {
        0.0: 1,  # Resuscitation → Critical
        1.0: 1,  # Critical → Critical
        2.0: 2,  # Emergent → Emergent
        3.0: 3,  # Urgent → Urgent
        4.0: 4,  # Less urgent → Standard
        5.0: 5,  # Non-urgent → Non-urgent
        7.0: 4,  # Unknown → Standard
    }
    
    # Reverse mapping: 5-class Severity → 7-class ESI (for interpretation)
    SEVERITY_TO_ESI = {
        1: [0.0, 1.0],  # Critical: ESI 0 or 1
        2: [2.0],       # Emergent: ESI 2
        3: [3.0],       # Urgent: ESI 3
        4: [4.0, 7.0],  # Standard: ESI 4 or 7
        5: [5.0],       # Non-urgent: ESI 5
    }
    
    # Severity labels
    SEVERITY_LABELS = {
        1: "Critical",
        2: "Emergent",
        3: "Urgent",
        4: "Standard",
        5: "Non-urgent"
    }
    
    def __init__(self):
        """Initialize ESI 5-class mapper."""
        pass
    
    def fit(self, y: pd.Series, X=None):
        """
        Fit mapper (no-op, mapping is deterministic).
        
        Args:
            y: Target Series with ESI levels
            X: Features (unused)
            
        Returns:
            self
        """
        # Check that all ESI levels are mappable
        unique_esi = y.unique()
        unmapped = [esi for esi in unique_esi if esi not in self.ESI_TO_SEVERITY]
        
        if unmapped:
            raise ValueError(f"Unmapped ESI levels found: {unmapped}")
        
        return self
    
    def transform(self, y: pd.Series) -> pd.Series:
        """
        Transform ESI levels to 5-class severity.
        
        Args:
            y: Series with ESI levels (0, 1, 2, 3, 4, 5, 7)
            
        Returns:
            Series with severity levels (1, 2, 3, 4, 5)
        """
        y_mapped = y.copy()
        
        # Map each ESI level to severity
        for esi_level, severity in self.ESI_TO_SEVERITY.items():
            y_mapped[y_mapped == esi_level] = severity
        
        return y_mapped.astype(int)
    
    def fit_transform(self, y: pd.Series, X=None) -> pd.Series:
        """Fit and transform in one step."""
        return self.fit(y, X).transform(y)
    
    def inverse_transform(self, y_severity: pd.Series) -> pd.Series:
        """
        Reverse transform: 5-class severity → 7-class ESI.
        
        Note: This is ambiguous (multiple ESI levels map to same severity).
        Returns the most common ESI level for each severity.
        
        Args:
            y_severity: Series with severity levels (1, 2, 3, 4, 5)
            
        Returns:
            Series with ESI levels (using most common ESI for each severity)
        """
        y_esi = y_severity.copy()
        
        # Use most common ESI for each severity (for reverse mapping)
        severity_to_most_common_esi = {
            1: 1.0,  # Critical: use ESI 1 (more common than 0)
            2: 2.0,  # Emergent: ESI 2
            3: 3.0,  # Urgent: ESI 3
            4: 4.0,  # Standard: use ESI 4 (more common than 7)
            5: 5.0,  # Non-urgent: ESI 5
        }
        
        for severity, esi_level in severity_to_most_common_esi.items():
            y_esi[y_esi == severity] = esi_level
        
        return y_esi.astype(float)
    
    @classmethod
    def get_distribution_comparison(cls, y_original: pd.Series) -> Dict:
        """
        Compare original 7-class vs mapped 5-class distribution.
        
        Args:
            y_original: Original ESI levels
            
        Returns:
            Dictionary with distribution comparison
        """
        mapper = cls()
        y_mapped = mapper.fit_transform(y_original)
        
        original_dist = y_original.value_counts(normalize=True).sort_index()
        mapped_dist = y_mapped.value_counts(normalize=True).sort_index()
        
        return {
            "original_7class": {
                "distribution": original_dist.to_dict(),
                "class_counts": y_original.value_counts().to_dict()
            },
            "mapped_5class": {
                "distribution": mapped_dist.to_dict(),
                "class_counts": y_mapped.value_counts().to_dict(),
                "labels": {sev: cls.SEVERITY_LABELS[sev] for sev in mapped_dist.index}
            }
        }

