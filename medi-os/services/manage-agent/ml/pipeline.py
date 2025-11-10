"""
Complete preprocessing pipeline for triage classification model.

Orchestrates all preprocessing steps in the correct order with proper
train/validation/test handling to prevent data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTENC
from sklearn.utils.class_weight import compute_class_weight
from typing import Tuple, Optional, List

from .data_splitter import RandomStratifiedSplitter
from .feature_engineering import DiagnosisDropper, CyclicalEncoder, OutlierClipper, RFVClusterEncoder
from .preprocessing import KNNImputerWrapper, YeoJohnsonTransformer
from .rfv_mapper import RFVTextToCodeMapper
from .rfv_sentence_embedder import RFVSentenceEmbedder
from .esi_5class_mapper import ESI5ClassMapper
from .clinical_feature_engineer import ClinicalFeatureEngineer


class TriagePreprocessingPipeline:
    """
    Complete preprocessing pipeline for NHAMCS triage classification.
    
    Applies all transformations in the correct order:
    1. Drop diagnosis columns (data leakage)
    2. Split data (train/val/test)
    3. KNN imputation (RFV codes treated as numeric)
    4. Outlier clipping
    5. Yeo-Johnson transformation
    6. Cyclical encoding (temporal)
    7. Standardization
    8. Class imbalance handling (SMOTE-NC + class weights)
    
    Note: RFV codes are kept as numeric throughout (no encoding needed for tree models)
    """
    
    def __init__(
        self,
        random_state: int = 42,
        knn_neighbors: int = 3,
        skew_threshold: float = 1.0,
        iqr_factor: float = 1.5,
        apply_smote: bool = True,
        knn_max_samples: Optional[int] = 50000,
        use_rfv_clustering: bool = False,
        use_rfv_embeddings: bool = True,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_pca: bool = True,
        embedding_pca_components: int = 20,
        use_5class_severity: bool = True
    ):
        """
        Initialize preprocessing pipeline.
        
        Args:
            random_state: Random seed for reproducibility
            knn_neighbors: Number of neighbors for KNN imputation (default: 3, reduced for speed)
            skew_threshold: Minimum skewness to trigger Yeo-Johnson
            iqr_factor: IQR multiplier for outlier clipping
            apply_smote: Whether to apply SMOTE-NC for class imbalance
            knn_max_samples: Max samples for KNN fitting (None = use all, for speed)
            use_rfv_clustering: If True, cluster RFV codes into 10-15 medical categories (default: False)
            use_rfv_embeddings: If True, use sentence embeddings for RFV (default: True)
            embedding_model: Sentence transformer model name
            embedding_pca: If True, reduce embeddings via PCA (default: True)
            embedding_pca_components: Number of PCA components if embedding_pca=True (default: 20)
            use_5class_severity: If True, map 7 ESI classes to 5 severity levels (default: True)
        """
        self.random_state = random_state
        self.knn_neighbors = knn_neighbors
        self.skew_threshold = skew_threshold
        self.iqr_factor = iqr_factor
        self.apply_smote = apply_smote
        self.use_rfv_clustering = use_rfv_clustering
        self.use_rfv_embeddings = use_rfv_embeddings
        self.use_5class_severity = use_5class_severity
        
        # ESI 5-class mapper
        self.esi_5class_mapper = ESI5ClassMapper() if use_5class_severity else None
        
        # Initialize components
        self.splitter = RandomStratifiedSplitter(random_state=random_state)
        self.diagnosis_dropper = DiagnosisDropper()
        
        # RFV processing: embeddings OR clustering (not both)
        if use_rfv_embeddings:
            self.rfv_embedder = RFVSentenceEmbedder(
                model_name=embedding_model,
                rfv_columns=['rfv1', 'rfv2'],
                use_pca=embedding_pca,
                pca_components=embedding_pca_components
            )
            self.rfv_cluster_encoder = None
        elif use_rfv_clustering:
            self.rfv_cluster_encoder = RFVClusterEncoder(
                rfv_columns=['rfv1', 'rfv2'],
                use_clustering=True
            )
            self.rfv_embedder = None
        else:
            self.rfv_embedder = None
            self.rfv_cluster_encoder = None
        
        self.knn_imputer = KNNImputerWrapper(
            n_neighbors=knn_neighbors,
            weights='uniform',  # Uniform is faster than distance
            max_samples=knn_max_samples
        )
        self.outlier_clipper = OutlierClipper(factor=iqr_factor)
        self.clinical_feature_engineer = ClinicalFeatureEngineer()
        self.yeo_johnson = YeoJohnsonTransformer(skew_threshold=skew_threshold)
        self.cyclical_encoder = CyclicalEncoder()
        self.scaler = StandardScaler()
        self.smote_nc = None
        
        # RFV mapper for inference (text → code conversion)
        self.rfv_mapper: Optional[RFVTextToCodeMapper] = None
        
        # Storage for fitted components
        self.feature_names_ = None
        self.class_weights_ = None
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str = 'esi_level',
        exclude_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Fit and transform complete pipeline on full dataset.
        
        Args:
            df: Full DataFrame with features and target
            target_col: Name of target column
            exclude_cols: Additional columns to exclude (e.g., 'year')
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("=" * 60)
        print("TRIAGE PREPROCESSING PIPELINE")
        print("=" * 60)
        
        # Extract features and target
        X_full = df.drop(columns=[target_col])
        y_full = df[target_col]
        
        # Step 1: Drop diagnosis columns (before splitting)
        print("\n[Step 1] Dropping diagnosis columns...")
        X_full = self.diagnosis_dropper.fit_transform(X_full, y_full)
        
        # Step 1.5: RFV Processing (before splitting)
        if self.use_rfv_embeddings and self.rfv_embedder:
            print("\n[Step 1.5] RFV Sentence Embeddings (converting RFV text to semantic embeddings)...")
            X_full = self.rfv_embedder.fit_transform(X_full)
            rfv_embed_cols = [col for col in X_full.columns if 'rfv' in col.lower() and 'emb' in col.lower()]
            print(f"  Created {len(rfv_embed_cols)} RFV embedding features")
        elif self.use_rfv_clustering and self.rfv_cluster_encoder:
            print("\n[Step 1.5] RFV Clustering (replacing 723 codes with 10-15 medical clusters)...")
            X_full = self.rfv_cluster_encoder.fit_transform(X_full)
            print(f"  RFV codes clustered into medical domain categories")
            rfv_cluster_cols = [col for col in X_full.columns if 'rfv' in col.lower() and 'cluster' in col.lower()]
            print(f"  Created {len(rfv_cluster_cols)} RFV cluster features")
        
        # Drop year column (used for splitting, not for features)
        if exclude_cols:
            for col in exclude_cols:
                if col in X_full.columns:
                    X_full = X_full.drop(columns=[col])
                    print(f"  Dropped {col} (excluded column)")
        
        # Step 2: Map ESI to 5-class severity if enabled
        if self.use_5class_severity and self.esi_5class_mapper:
            print("\n[Step 2] Mapping ESI to 5-class severity...")
            y_full_mapped = self.esi_5class_mapper.fit_transform(y_full)
            
            # Show distribution comparison
            dist_comparison = self.esi_5class_mapper.get_distribution_comparison(y_full)
            print(f"  Original 7-class distribution:")
            for esi, pct in dist_comparison["original_7class"]["distribution"].items():
                print(f"    ESI {esi}: {pct*100:.1f}%")
            print(f"  Mapped 5-class distribution:")
            for sev, pct in dist_comparison["mapped_5class"]["distribution"].items():
                label = dist_comparison["mapped_5class"]["labels"][sev]
                print(f"    Severity {sev} ({label}): {pct*100:.1f}%")
            
            y_full = y_full_mapped
        
        # Step 3: Split data (before any other preprocessing)
        print("\n[Step 3] Splitting data (70/15/15)...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.splitter.split(
            X_full, y_full
        )
        
        # Step 4: Load RFV mapper for inference (if mappings exist)
        try:
            self.rfv_mapper = RFVTextToCodeMapper.load()
            print(f"\n[Step 4] Loaded RFV text-to-code mapper for inference")
        except FileNotFoundError:
            print(f"\n[Step 4] Warning: RFV mappings not found. Inference text conversion will not work.")
            print(f"  Run parser to generate rfv_code_mappings.json")
        
        # Step 5: KNN Imputation
        print("\n[Step 5] KNN Imputation (k={})...".format(self.knn_neighbors))
        print("  Note: This step may take 5-15 minutes due to distance calculations.")
        print("  Processing in chunks to show progress...")
        # Use chunks for training data too (it's large: 125k rows)
        X_train = self.knn_imputer.fit_transform(X_train, chunk_size=5000)  # Smaller chunks = more frequent progress
        print("\n  Processing validation set...")
        X_val = self.knn_imputer.transform(X_val, chunk_size=5000)
        print("\n  Processing test set...")
        X_test = self.knn_imputer.transform(X_test, chunk_size=5000)
        
        # Step 5: Outlier Clipping
        print("\n[Step 5] Outlier Clipping (IQR factor={})...".format(self.iqr_factor))
        X_train = self.outlier_clipper.fit_transform(X_train)
        X_val = self.outlier_clipper.transform(X_val)
        X_test = self.outlier_clipper.transform(X_test)
        
        # Step 5.5: Clinical Feature Engineering
        # (after Step 5: Outlier Clipping)
        # (before Step 7: Yeo-Johnson)
        print("\n[Step 5.5] Clinical Feature Engineering...")
        X_train = self.clinical_feature_engineer.fit_transform(X_train)
        X_val = self.clinical_feature_engineer.transform(X_val)
        X_test = self.clinical_feature_engineer.transform(X_test)
        
        # Step 7: Yeo-Johnson Transformation
        print("\n[Step 7] Yeo-Johnson Transformation (skew threshold={})...".format(self.skew_threshold))
        X_train = self.yeo_johnson.fit_transform(X_train)
        X_val = self.yeo_johnson.transform(X_val)
        X_test = self.yeo_johnson.transform(X_test)
        
        # Step 8: Cyclical Encoding (temporal features)
        print("\n[Step 8] Cyclical Encoding (temporal features)...")
        X_train = self.cyclical_encoder.fit_transform(X_train)
        X_val = self.cyclical_encoder.transform(X_val)
        X_test = self.cyclical_encoder.transform(X_test)
        
        # Step 9: Standardization
        print("\n[Step 9] Standardization...")
        X_train = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_val = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        X_test = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Store feature names
        self.feature_names_ = X_train.columns.tolist()
        print(f"  Final feature count: {len(self.feature_names_)}")
        if self.use_rfv_embeddings:
            print(f"  Note: RFV codes converted to sentence embeddings")
        elif self.use_rfv_clustering:
            print(f"  Note: RFV codes clustered into medical domain categories (one-hot encoded)")
        else:
            print(f"  Note: RFV codes kept as numeric")
        
        # Step 10: Class Imbalance Handling (SMOTE-NC + Class Weights)
        if self.apply_smote:
            print("\n[Step 10] Class Imbalance Handling...")
            X_train, y_train = self._apply_smote_nc(X_train, y_train)
        
        # Calculate class weights
        self.class_weights_ = self._calculate_class_weights(y_train)
        
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE")
        print("=" * 60)
        print(f"Train shape: {X_train.shape}")
        print(f"Validation shape: {X_val.shape}")
        print(f"Test shape: {X_test.shape}")
        print(f"Features: {len(self.feature_names_)}")
        print(f"Classes: {sorted(y_train.unique())}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _apply_smote_nc(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE-NC (SMOTE for Nominal and Continuous) to handle class imbalance.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Resampled (X_train, y_train)
        """
        # Identify categorical (binary) feature indices
        categorical_indices = []
        for i, col in enumerate(X_train.columns):
            # Binary features (0/1 only)
            if X_train[col].nunique() == 2 and set(X_train[col].unique()) == {0, 1}:
                categorical_indices.append(i)
        
        if not categorical_indices:
            print("  Warning: No categorical features detected for SMOTE-NC")
            print("  Using standard SMOTE...")
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=self.random_state)
        else:
            print(f"  SMOTE-NC: {len(categorical_indices)} categorical features detected")
            self.smote_nc = SMOTENC(
                categorical_features=categorical_indices,
                random_state=self.random_state
            )
            smote = self.smote_nc
        
        # Apply SMOTE
        print(f"  Original class distribution:")
        print(y_train.value_counts().sort_index())
        
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"  After SMOTE-NC:")
        y_resampled_series = pd.Series(y_resampled)
        print(y_resampled_series.value_counts().sort_index())
        print(f"  Samples before: {len(X_train)}, after: {len(X_resampled)}")
        
        # Convert back to DataFrame
        X_resampled_df = pd.DataFrame(
            X_resampled,
            columns=X_train.columns,
            index=range(len(X_resampled))
        )
        
        return X_resampled_df, y_resampled_series
    
    def _calculate_class_weights(self, y: pd.Series) -> dict:
        """
        Calculate class weights for imbalanced classes.
        
        Args:
            y: Target Series
            
        Returns:
            Dictionary mapping class to weight
        """
        classes = np.unique(y)
        weights = compute_class_weight(
            'balanced',
            classes=classes,
            y=y.values
        )
        
        class_weights = dict(zip(classes, weights))
        
        print("\nClass weights:")
        for cls, weight in sorted(class_weights.items()):
            count = (y == cls).sum()
            print(f"  ESI {int(cls)}: {weight:.3f} (n={count:,})")
        
        return class_weights
    
    def transform_new_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted pipeline.
        
        For inference/deployment on new patient data.
        Must apply transformations in the SAME ORDER as fit_transform.
        
        Args:
            X: New feature DataFrame
            
        Returns:
            Transformed DataFrame
        """
        # Apply all transformations in order (fitted components)
        # Same order as fit_transform
        X = self.diagnosis_dropper.transform(X)
        
        # Apply RFV processing if enabled
        if self.use_rfv_embeddings and self.rfv_embedder:
            X = self.rfv_embedder.transform(X)
        elif self.use_rfv_clustering and self.rfv_cluster_encoder:
            X = self.rfv_cluster_encoder.transform(X)
        
        X = self.knn_imputer.transform(X)
        X = self.outlier_clipper.transform(X)
        X = self.clinical_feature_engineer.transform(X)
        X = self.yeo_johnson.transform(X)
        X = self.cyclical_encoder.transform(X)
        X = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        return X
    
    def preprocess_new_patient(
        self,
        patient_data: dict,
        convert_rfv_text: bool = True
    ) -> np.ndarray:
        """
        Preprocess a new patient's data for inference.
        
        Handles text RFV inputs by converting them to numeric codes,
        then runs through the preprocessing pipeline.
        
        Args:
            patient_data: Dictionary with patient features. RFV can be:
                         - Text: {"rfv1": "Chest pain", ...}
                         - Numeric: {"rfv1": 10501.0, ...}
            convert_rfv_text: If True, convert text RFV to codes using mapper
            
        Returns:
            Preprocessed feature array ready for model prediction
            
        Example:
            patient = {
                "age": 45,
                "sex": 1,
                "temp_c": 38.5,
                "pulse": 85,
                "rfv1": "Chest pain",  # Text input
                "rfv2": "Shortness of breath",
                ...
            }
            features = pipeline.preprocess_new_patient(patient)
            prediction = model.predict(features)
        """
        # Convert to DataFrame
        patient_df = pd.DataFrame([patient_data])
        
        # Convert text RFV to codes if needed
        if convert_rfv_text and self.rfv_mapper:
            rfv_fields = ['rfv1', 'rfv2', 'rfv3', 'rfv1_3d', 'rfv2_3d', 'rfv3_3d']
            
            for field in rfv_fields:
                if field in patient_df.columns:
                    value = patient_df[field].iloc[0]
                    
                    # Check if it's text (string) and needs conversion
                    if isinstance(value, str) and value:
                        code = self.rfv_mapper.text_to_code(value, field)
                        if code is not None:
                            patient_df[field] = code
                        else:
                            # If not found, use most common code for this field
                            # or set to 0 (missing/unknown)
                            patient_df[field] = 0.0
                            print(f"Warning: Could not map '{value}' to code for {field}, using 0")
        
        # Run through preprocessing pipeline
        preprocessed = self.transform_new_data(patient_df)
        
        # Ensure feature order matches training
        if self.feature_names_:
            preprocessed = preprocessed[self.feature_names_]
        
        return preprocessed.values


def create_preprocessing_pipeline(
    random_state: int = 42,
    **kwargs
) -> TriagePreprocessingPipeline:
    """
    Factory function to create preprocessing pipeline with default settings.
    
    Args:
        random_state: Random seed
        **kwargs: Additional arguments for pipeline
        
    Returns:
        Configured TriagePreprocessingPipeline
    """
    return TriagePreprocessingPipeline(random_state=random_state, **kwargs)

