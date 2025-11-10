"""
Feature Engineering Components

Custom transformers for feature engineering steps in the preprocessing pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'services' / 'manage-agent'))

from ml.rfv_clustering import RFVClusterer


class DiagnosisDropper(BaseEstimator, TransformerMixin):
    """
    Drops diagnosis columns to prevent data leakage.
    
    Diagnosis is determined AFTER triage, so it cannot be used
    to predict ESI level (circular dependency).
    """
    
    def __init__(self):
        """Initialize diagnosis dropper."""
        self.diagnosis_cols_ = None
    
    def fit(self, X: pd.DataFrame, y=None):
        """Identify diagnosis columns."""
        diagnosis_keywords = ['diag', 'diagnosis']
        self.diagnosis_cols_ = [
            col for col in X.columns
            if any(keyword in col.lower() for keyword in diagnosis_keywords)
        ]
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop diagnosis columns."""
        if self.diagnosis_cols_:
            X = X.drop(columns=self.diagnosis_cols_, errors='ignore')
        return X


class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes cyclical features (month, day_of_week) using sine/cosine.
    
    This preserves the cyclical nature (e.g., Dec is close to Jan).
    """
    
    def __init__(self):
        """Initialize cyclical encoder."""
        self.cyclical_cols_ = None
    
    def fit(self, X: pd.DataFrame, y=None):
        """Identify cyclical columns."""
        self.cyclical_cols_ = []
        for col in ['month', 'day_of_week']:
            if col in X.columns:
                self.cyclical_cols_.append(col)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode cyclical features."""
        X = X.copy()
        
        for col in self.cyclical_cols_:
            if col not in X.columns:
                continue
            
            # Get max value for scaling
            max_val = X[col].max()
            if max_val > 0:
                # Normalize to [0, 2π] range
                normalized = 2 * np.pi * X[col] / max_val
                
                # Create sin and cos columns
                X[f'{col}_sin'] = np.sin(normalized)
                X[f'{col}_cos'] = np.cos(normalized)
            
            # Drop original column
            X = X.drop(columns=[col])
        
        return X


class RFVClusterEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes RFV codes into clusters and one-hot encodes clusters.
    
    Replaces 723 RFV codes with 13-15 medical domain clusters,
    then one-hot encodes the clusters for use in ML models.
    """
    
    def __init__(self, rfv_columns: Optional[List[str]] = None, use_clustering: bool = True):
        """
        Initialize RFV cluster encoder.
        
        Args:
            rfv_columns: List of RFV column names (default: ['rfv1', 'rfv2'])
            use_clustering: If True, use clustering; if False, keep numeric codes
        """
        self.rfv_columns = rfv_columns or ['rfv1', 'rfv2']
        self.use_clustering = use_clustering
        self.clusterer: Optional[RFVClusterer] = None
        self.cluster_columns_ = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit RFV clusterer."""
        if self.use_clustering:
            # Create clusterer
            self.clusterer = RFVClusterer()
            
            # Determine which clusters will be created
            # Get all unique clusters from sample data
            sample_clusters = set()
            for rfv_col in self.rfv_columns:
                if rfv_col in X.columns:
                    df_temp = self.clusterer.fit_transform(X[[rfv_col]], rfv_col)
                    cluster_col = f"{rfv_col}_cluster"
                    sample_clusters.update(df_temp[cluster_col].unique())
            
            # Store cluster names (will be used for one-hot encoding)
            self.cluster_columns_ = sorted(list(sample_clusters))
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform RFV codes to clustered one-hot features."""
        X = X.copy()
        
        if not self.use_clustering:
            # Keep RFV codes as numeric (for comparison)
            return X
        
        if self.clusterer is None:
            raise ValueError("RFVClusterEncoder not fitted. Call fit() first.")
        
        # Process each RFV column
        for rfv_col in self.rfv_columns:
            if rfv_col not in X.columns:
                continue
            
            # Convert codes to clusters
            df_temp = self.clusterer.fit_transform(X[[rfv_col]], rfv_col)
            cluster_col = f"{rfv_col}_cluster"
            clusters = df_temp[cluster_col]
            
            # One-hot encode clusters
            cluster_onehot = pd.get_dummies(
                clusters,
                prefix=f"{rfv_col}_cluster",
                prefix_sep="_",
                dummy_na=False
            )
            
            # Ensure all expected clusters are present (fill missing with 0)
            for cluster_name in self.cluster_columns_:
                expected_col = f"{rfv_col}_cluster_{cluster_name}"
                if expected_col not in cluster_onehot.columns:
                    cluster_onehot[expected_col] = 0
            
            # Drop original RFV column and add one-hot encoded clusters
            X = X.drop(columns=[rfv_col])
            X = pd.concat([X, cluster_onehot], axis=1)
        
        return X


class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Clips outliers using IQR method.
    
    Values beyond Q1 - factor*IQR or Q3 + factor*IQR are clipped.
    """
    
    def __init__(self, factor: float = 1.5):
        """
        Initialize outlier clipper.
        
        Args:
            factor: IQR multiplier (default: 1.5)
        """
        self.factor = factor
        self.clip_bounds_ = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        """Compute clipping bounds for numeric columns."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - self.factor * iqr
            upper_bound = q3 + self.factor * iqr
            
            self.clip_bounds_[col] = (lower_bound, upper_bound)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clip outliers."""
        X = X.copy()
        
        for col, (lower, upper) in self.clip_bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=lower, upper=upper)
        
        return X
