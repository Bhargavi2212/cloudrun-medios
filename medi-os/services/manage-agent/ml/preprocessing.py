"""
Preprocessing transformers for clinical triage data.

Includes KNN imputation, Yeo-Johnson transformation, and target encoding
for RFV categorical features.
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional, Dict
import time


class KNNImputerWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper for KNN imputation with train-only fitting.
    
    Uses k nearest neighbors to impute missing values based on
    similar patients, preserving relationships between variables.
    
    Optimized for large datasets with reduced neighbors and progress tracking.
    """
    
    def __init__(self, n_neighbors: int = 3, weights: str = 'uniform', max_samples: Optional[int] = 50000):
        """
        Initialize KNN imputer.
        
        Args:
            n_neighbors: Number of neighbors to use (default: 3, reduced for speed)
            weights: Weight function ('uniform' or 'distance', uniform is faster)
            max_samples: Maximum samples to use for fitting (None = use all, for very large datasets)
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.max_samples = max_samples
        self.imputer = None
        self.feature_names_ = None
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit imputer on training data.
        
        Args:
            X: Training DataFrame
            y: Target (ignored, for sklearn compatibility)
        """
        print(f"  Fitting KNN imputer (k={self.n_neighbors}, samples={len(X):,})...")
        
        # Sample if max_samples specified (for very large datasets)
        if self.max_samples and len(X) > self.max_samples:
            print(f"  Sampling {self.max_samples:,} rows for faster fitting...")
            X_fit = X.sample(n=self.max_samples, random_state=42)
        else:
            X_fit = X
        
        self.imputer = KNNImputer(
            n_neighbors=self.n_neighbors,
            weights=self.weights
        )
        print("  Computing nearest neighbors (this may take a few minutes)...")
        start_time = time.time()
        self.imputer.fit(X_fit.values)
        fit_time = time.time() - start_time
        self.feature_names_ = X.columns.tolist()
        print(f"  KNN imputer fitted successfully (took {fit_time:.1f} seconds)")
        return self
    
    def transform(self, X: pd.DataFrame, chunk_size: int = 10000) -> pd.DataFrame:
        """
        Transform data by imputing missing values in chunks for efficiency.
        
        Args:
            X: DataFrame to transform
            chunk_size: Number of rows to process at a time (default: 10000)
            
        Returns:
            DataFrame with imputed values
        """
        if self.imputer is None:
            raise ValueError("Must fit transformer before transforming")
        
        total_rows = len(X)
        print(f"  Imputing missing values ({total_rows:,} rows) in chunks of {chunk_size:,}...")
        
        # Process in chunks to save memory and show progress
        if total_rows <= chunk_size:
            # Small dataset, process all at once
            X_imputed = self.imputer.transform(X.values)
            return pd.DataFrame(X_imputed, columns=self.feature_names_, index=X.index)
        
        # Process in chunks
        chunks = []
        num_chunks = (total_rows + chunk_size - 1) // chunk_size
        start_time = time.time()
        
        for i in range(0, total_rows, chunk_size):
            chunk_end = min(i + chunk_size, total_rows)
            chunk = X.iloc[i:chunk_end]
            
            chunk_num = (i // chunk_size) + 1
            chunk_start_time = time.time()
            
            print(f"    Chunk {chunk_num}/{num_chunks} (rows {i:,}-{chunk_end:,})...", end="", flush=True)
            
            # This is the slow part - sklearn's internal distance calculations
            chunk_imputed = self.imputer.transform(chunk.values)
            chunks.append(chunk_imputed)
            
            chunk_time = time.time() - chunk_start_time
            elapsed = time.time() - start_time
            avg_time_per_chunk = elapsed / chunk_num
            remaining_chunks = num_chunks - chunk_num
            eta = avg_time_per_chunk * remaining_chunks
            
            print(f" {chunk_time:.1f}s (avg: {avg_time_per_chunk:.1f}s/chunk, ETA: {eta/60:.1f} min)")
        
        # Combine all chunks
        X_imputed = np.vstack(chunks)
        total_time = time.time() - start_time
        print(f"  Imputation complete (total: {total_time/60:.1f} minutes)")
        return pd.DataFrame(X_imputed, columns=self.feature_names_, index=X.index)
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, chunk_size: int = 10000) -> pd.DataFrame:
        """Fit and transform in one step, with chunking for transform."""
        return self.fit(X, y).transform(X, chunk_size=chunk_size)


class YeoJohnsonTransformer(BaseEstimator, TransformerMixin):
    """
    Yeo-Johnson power transformation for handling skewness.
    
    Handles zeros and negative values (unlike Box-Cox).
    Only transforms columns with high skewness (|skew| > threshold).
    """
    
    def __init__(self, skew_threshold: float = 1.0, standardize: bool = False):
        """
        Initialize Yeo-Johnson transformer.
        
        Args:
            skew_threshold: Minimum absolute skewness to trigger transformation (default: 1.0)
            standardize: Whether to standardize after transformation (default: False)
        """
        self.skew_threshold = skew_threshold
        self.standardize = standardize
        self.transformers: Dict[str, PowerTransformer] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.columns_to_transform: List[str] = []
        self.feature_names_ = None
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit transformers on training data.
        
        Only fits transformers for columns with high skewness.
        
        Args:
            X: Training DataFrame
            y: Target (ignored)
        """
        self.feature_names_ = X.columns.tolist()
        
        # Identify columns to transform (high skewness)
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                skewness = X[col].skew()
                if abs(skewness) > self.skew_threshold:
                    self.columns_to_transform.append(col)
                    
                    # Create transformer for this column
                    transformer = PowerTransformer(
                        method='yeo-johnson',
                        standardize=False  # We'll standardize separately if needed
                    )
                    transformer.fit(X[[col]])
                    self.transformers[col] = transformer
                    
                    # Create scaler if standardization requested
                    if self.standardize:
                        scaler = StandardScaler()
                        # Fit on transformed data
                        transformed = transformer.transform(X[[col]])
                        scaler.fit(transformed)
                        self.scalers[col] = scaler
        
        print(f"Yeo-Johnson: Transforming {len(self.columns_to_transform)} columns "
              f"with |skew| > {self.skew_threshold}")
        if self.columns_to_transform:
            print(f"  Columns: {', '.join(self.columns_to_transform)}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by applying Yeo-Johnson to skewed columns.
        
        Args:
            X: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        if not self.transformers:
            return X.copy()
        
        X_transformed = X.copy()
        
        for col in self.columns_to_transform:
            if col in self.transformers:
                # Transform
                transformed = self.transformers[col].transform(X[[col]])
                
                # Standardize if requested
                if self.standardize and col in self.scalers:
                    transformed = self.scalers[col].transform(transformed)
                
                X_transformed[col] = transformed
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target encoder (mean encoding) for categorical features.
    
    Encodes categorical values as the mean target value for that category.
    This is ideal for RFV (Reason for Visit) columns which are categorical
    but stored as text.
    """
    
    def __init__(
        self,
        columns: Optional[List[str]] = None,
        smoothing: float = 1.0,
        handle_unknown: str = 'mean'
    ):
        """
        Initialize target encoder.
        
        Args:
            columns: List of column names to encode. If None, auto-detect categorical columns.
            smoothing: Smoothing parameter to prevent overfitting (higher = more smoothing)
            handle_unknown: How to handle unseen categories ('mean', 'ignore', or 'error')
        """
        self.columns = columns
        self.smoothing = smoothing
        self.handle_unknown = handle_unknown
        self.encodings: Dict[str, Dict[str, float]] = {}
        self.global_means: Dict[str, float] = {}
        self.feature_names_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit encoder on training data.
        
        IMPORTANT: Must be fit only on training data to prevent data leakage.
        
        Args:
            X: Training DataFrame
            y: Training target Series
        """
        if self.columns is None:
            # Auto-detect categorical columns (object dtype or low cardinality)
            self.columns = []
            for col in X.columns:
                if X[col].dtype == 'object' or X[col].nunique() < 100:
                    # Check if it's an RFV column
                    if 'rfv' in col.lower():
                        self.columns.append(col)
        
        self.feature_names_ = []
        
        for col in self.columns:
            if col not in X.columns:
                continue
            
            # Combine X[col] and y to calculate mean target per category
            temp_df = pd.DataFrame({col: X[col], 'target': y.values})
            
            # Calculate mean target per category
            category_means = temp_df.groupby(col)['target'].agg('mean')
            
            # Global mean for smoothing
            global_mean = y.mean()
            
            # Calculate counts per category
            category_counts = X[col].value_counts()
            
            # Apply smoothing: weighted average between category mean and global mean
            encoding = {}
            for category in category_means.index:
                count = category_counts[category]
                # Smoothing: more data → trust category mean more
                weight = count / (count + self.smoothing)
                smoothed_mean = weight * category_means[category] + (1 - weight) * global_mean
                encoding[category] = smoothed_mean
            
            self.encodings[col] = encoding
            self.global_means[col] = global_mean
            
            # Create new feature name
            self.feature_names_.append(f"{col}_encoded")
        
        print(f"Target encoding: Encoded {len(self.columns)} columns")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by replacing categories with encoded values.
        
        Args:
            X: DataFrame to transform
            
        Returns:
            DataFrame with encoded columns (original columns removed)
        """
        if not self.encodings:
            raise ValueError("Must fit encoder before transforming")
        
        X_encoded = X.copy()
        
        for col in self.columns:
            if col not in X.columns:
                continue
            
            new_col_name = f"{col}_encoded"
            
            # Map categories to encoded values
            encoded_values = X[col].map(self.encodings[col])
            
            # Handle unseen categories
            if encoded_values.isna().any():
                if self.handle_unknown == 'mean':
                    encoded_values = encoded_values.fillna(self.global_means[col])
                elif self.handle_unknown == 'ignore':
                    encoded_values = encoded_values.fillna(0)  # Default to 0
                elif self.handle_unknown == 'error':
                    unseen = X[col][encoded_values.isna()].unique()
                    raise ValueError(f"Unseen categories in {col}: {unseen}")
            
            # Replace column with encoded version
            X_encoded[new_col_name] = encoded_values
            X_encoded = X_encoded.drop(columns=[col])
        
        return X_encoded
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


class RFVEncoder(BaseEstimator, TransformerMixin):
    """
    One-hot encoder for RFV numeric codes.
    
    Encodes RFV (Reason for Visit) numeric codes using one-hot encoding
    for top N most frequent codes. More efficient than text target encoding.
    """
    
    def __init__(
        self,
        columns: Optional[List[str]] = None,
        top_n: int = 50,
        handle_unknown: str = 'other'
    ):
        """
        Initialize RFV encoder.
        
        Args:
            columns: List of RFV column names to encode. If None, auto-detect.
            top_n: Number of top frequent codes to keep as separate features (default: 50)
            handle_unknown: How to handle infrequent codes ('other' or 'ignore')
        """
        self.columns = columns
        self.top_n = top_n
        self.handle_unknown = handle_unknown
        self.top_codes: Dict[str, List[float]] = {}
        self.feature_names_ = []
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit encoder on training data.
        
        Identifies top N most frequent codes for each RFV column.
        
        Args:
            X: Training DataFrame
            y: Target (ignored, for sklearn compatibility)
        """
        if self.columns is None:
            # Auto-detect RFV columns
            self.columns = [col for col in X.columns if col.startswith('rfv')]
        
        self.feature_names_ = []
        
        for col in self.columns:
            if col not in X.columns:
                continue
            
            # Get top N most frequent codes
            value_counts = X[col].value_counts()
            top_codes_list = value_counts.head(self.top_n).index.tolist()
            self.top_codes[col] = top_codes_list
            
            # Create feature names for one-hot encoding
            for code in top_codes_list:
                self.feature_names_.append(f"{col}_{int(code)}")
            
            # Add "other" category if handling unknown
            if self.handle_unknown == 'other':
                self.feature_names_.append(f"{col}_other")
        
        print(f"RFV encoding: One-hot encoding {len(self.columns)} columns")
        print(f"  Total features: {len(self.feature_names_)}")
        for col in self.columns:
            if col in self.top_codes:
                print(f"  {col}: Top {len(self.top_codes[col])} codes + other")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by one-hot encoding RFV codes.
        
        Args:
            X: DataFrame to transform
            
        Returns:
            DataFrame with one-hot encoded RFV columns (original columns removed)
        """
        if not self.top_codes:
            raise ValueError("Must fit encoder before transforming")
        
        X_encoded = X.copy()
        
        for col in self.columns:
            if col not in X.columns:
                continue
            
            # Create one-hot encoded columns
            for code in self.top_codes[col]:
                feature_name = f"{col}_{int(code)}"
                X_encoded[feature_name] = (X[col] == code).astype(int)
            
            # Handle "other" category (codes not in top N)
            if self.handle_unknown == 'other':
                other_feature = f"{col}_other"
                is_top_code = X[col].isin(self.top_codes[col])
                X_encoded[other_feature] = (~is_top_code).astype(int)
            
            # Drop original RFV column
            X_encoded = X_encoded.drop(columns=[col])
        
        return X_encoded
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

