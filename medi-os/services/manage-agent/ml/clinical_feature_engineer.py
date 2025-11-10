"""
Clinical Feature Engineering

Creates clinically meaningful derived features for triage classification:
- Clinical ratios (shock_index, MAP, pulse_pressure)
- Binary clinical thresholds (tachycardia, bradycardia, etc.)
- Age risk categories (elderly, pediatric, infant)
- Pain severity flags
- Time-based flags
- Aggregate/interaction features
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates 22 clinically meaningful derived features.
    
    Features include:
    - Clinical ratios (shock_index, MAP, pulse_pressure)
    - Binary clinical thresholds (8 features)
    - Age risk categories (3 features)
    - Pain severity flags (2 features)
    - Time-based flags (2 features)
    - Aggregate/interaction features (4 features)
    """
    
    def __init__(self):
        """Initialize clinical feature engineer."""
        self.new_features_ = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Validate required columns exist.
        
        Args:
            X: DataFrame with required columns
            y: Not used (for sklearn compatibility)
            
        Returns:
            self
        """
        required = ['pulse', 'sbp', 'dbp', 'o2_sat', 'respiration', 'age', 'pain', 'temp_c']
        missing = [col for col in required if col not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create clinical derived features.
        
        Args:
            X: DataFrame with original features
            
        Returns:
            DataFrame with added clinical features
        """
        X = X.copy()
        
        # Track new features for validation
        self.new_features_ = []
        
        # ========================================================================
        # 1. CLINICAL DERIVED FEATURES (3 features)
        # ========================================================================
        
        # Shock Index: pulse / sbp (default 0.7 if missing to avoid NaN)
        X['shock_index'] = np.where(
            (X['sbp'] > 0) & (X['pulse'] > 0) & 
            X['sbp'].notna() & X['pulse'].notna(),
            X['pulse'] / X['sbp'],
            0.7  # Normal shock index default
        )
        self.new_features_.append('shock_index')
        
        # Mean Arterial Pressure (MAP)
        X['mean_arterial_pressure'] = (2 * X['dbp'] + X['sbp']) / 3
        self.new_features_.append('mean_arterial_pressure')
        
        # Pulse Pressure
        X['pulse_pressure'] = X['sbp'] - X['dbp']
        self.new_features_.append('pulse_pressure')
        
        # ========================================================================
        # 2. CLINICAL THRESHOLD FLAGS (8 features)
        # ========================================================================
        
        # Heart rate flags
        X['tachycardia'] = (X['pulse'] > 120).fillna(0).astype(int)
        self.new_features_.append('tachycardia')
        
        X['bradycardia'] = (X['pulse'] < 50).fillna(0).astype(int)
        self.new_features_.append('bradycardia')
        
        # Blood pressure flags
        X['hypotension'] = (X['sbp'] < 90).fillna(0).astype(int)
        self.new_features_.append('hypotension')
        
        X['hypertension'] = (X['sbp'] > 180).fillna(0).astype(int)
        self.new_features_.append('hypertension')
        
        # Respiratory flags
        X['tachypnea'] = (X['respiration'] > 24).fillna(0).astype(int)
        self.new_features_.append('tachypnea')
        
        X['respiratory_distress'] = ((X['respiration'] > 24) | (X['o2_sat'] < 92)).fillna(0).astype(int)
        self.new_features_.append('respiratory_distress')
        
        X['hypoxia'] = (X['o2_sat'] < 92).fillna(0).astype(int)
        self.new_features_.append('hypoxia')
        
        X['severe_hypoxia'] = (X['o2_sat'] < 88).fillna(0).astype(int)
        self.new_features_.append('severe_hypoxia')
        
        # ========================================================================
        # 3. AGE RISK CATEGORIES (3 features)
        # ========================================================================
        
        X['is_elderly'] = (X['age'] > 65).fillna(0).astype(int)
        self.new_features_.append('is_elderly')
        
        X['is_pediatric'] = (X['age'] < 18).fillna(0).astype(int)
        self.new_features_.append('is_pediatric')
        
        X['is_infant'] = (X['age'] < 2).fillna(0).astype(int)
        self.new_features_.append('is_infant')
        
        # ========================================================================
        # 4. PAIN SEVERITY FLAGS (2 features)
        # ========================================================================
        
        X['severe_pain'] = (X['pain'] >= 8).fillna(0).astype(int)
        self.new_features_.append('severe_pain')
        
        X['moderate_pain'] = ((X['pain'] >= 5) & (X['pain'] < 8)).fillna(0).astype(int)
        self.new_features_.append('moderate_pain')
        
        # ========================================================================
        # 5. TIME-BASED FEATURES (2 features, graceful handling if missing)
        # ========================================================================
        
        if 'month' in X.columns:
            X['is_flu_season'] = X['month'].isin([11, 12, 1, 2, 3]).fillna(0).astype(int)
            self.new_features_.append('is_flu_season')
        
        if 'day_of_week' in X.columns:
            X['is_weekend'] = X['day_of_week'].isin([5, 6]).fillna(0).astype(int)
            self.new_features_.append('is_weekend')
        
        # ========================================================================
        # 6. AGGREGATE/INTERACTION FEATURES (4 features)
        # ========================================================================
        
        # Vital abnormality count
        X['vital_abnormal_count'] = (
            X['tachycardia'] + X['bradycardia'] + 
            X['hypotension'] + X['tachypnea'] + 
            X['hypoxia']
        ).astype(int)
        self.new_features_.append('vital_abnormal_count')
        
        # Shock index high threshold
        X['shock_index_high'] = (X['shock_index'] > 0.9).fillna(0).astype(int)
        self.new_features_.append('shock_index_high')
        
        # Elderly hypotension interaction
        X['elderly_hypotension'] = (X['is_elderly'] * X['hypotension']).astype(int)
        self.new_features_.append('elderly_hypotension')
        
        # Pediatric fever interaction
        X['pediatric_fever'] = (X['is_pediatric'] * (X['temp_c'] > 38.5)).fillna(0).astype(int)
        self.new_features_.append('pediatric_fever')
        
        # ========================================================================
        # FINAL VALIDATION: Replace Inf/NaN with 0 for all new features
        # ========================================================================
        
        for feat in self.new_features_:
            if feat in X.columns:
                X[feat] = X[feat].replace([np.inf, -np.inf], 0).fillna(0)
        
        return X

