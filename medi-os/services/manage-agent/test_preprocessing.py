"""
Test script for preprocessing pipeline.

Tests the complete preprocessing pipeline on NHAMCS dataset.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'services' / 'manage-agent'))

import pandas as pd
from ml.pipeline import TriagePreprocessingPipeline

def test_preprocessing_pipeline():
    """Test the complete preprocessing pipeline."""
    print("Loading data...")
    
    # Load combined dataset
    data_path = project_root / "data" / "NHAMCS_2011_2022_combined.csv"
    df = pd.read_csv(data_path)
    
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Initialize pipeline
    pipeline = TriagePreprocessingPipeline(
        random_state=42,
        knn_neighbors=5,
        skew_threshold=1.0,
        apply_smote=True
    )
    
    # Run preprocessing
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.fit_transform(
        df,
        target_col='esi_level',
        exclude_cols=['year']  # Exclude year (used for splitting in temporal, not needed for random)
    )
    
    # Validation checks
    print("\n" + "=" * 60)
    print("VALIDATION CHECKS")
    print("=" * 60)
    
    # Check 1: No missing values
    print("\n1. Checking for missing values...")
    train_missing = X_train.isnull().sum().sum()
    val_missing = X_val.isnull().sum().sum()
    test_missing = X_test.isnull().sum().sum()
    
    if train_missing == 0 and val_missing == 0 and test_missing == 0:
        print("   ✓ No missing values in any split")
    else:
        print(f"   ✗ Missing values found: Train={train_missing}, Val={val_missing}, Test={test_missing}")
    
    # Check 2: Diagnosis columns removed
    print("\n2. Checking diagnosis columns removed...")
    diag_cols = [col for col in X_train.columns if 'diag' in col.lower()]
    if len(diag_cols) == 0:
        print("   ✓ All diagnosis columns removed")
    else:
        print(f"   ✗ Diagnosis columns still present: {diag_cols}")
    
    # Check 3: RFV columns one-hot encoded
    print("\n3. Checking RFV one-hot encoding...")
    rfv_encoded_cols = [col for col in X_train.columns if 'rfv' in col.lower() and ('_' in col or col.startswith('rfv'))]
    if len(rfv_encoded_cols) > 0:
        print(f"   ✓ RFV columns one-hot encoded: {len(rfv_encoded_cols)} binary features")
        print(f"     Sample encoded columns: {rfv_encoded_cols[:10]}...")
        # Check if they're binary (0/1)
        sample_rfv = X_train[rfv_encoded_cols[:5]]
        is_binary = all(sample_rfv[col].isin([0, 1]).all() for col in sample_rfv.columns)
        if is_binary:
            print(f"     ✓ All RFV features are binary (0/1)")
        else:
            print(f"     ⚠ Some RFV features are not binary")
    else:
        print("   ✗ No RFV encoded columns found")
    
    # Check 4: Cyclical encoding applied
    print("\n4. Checking cyclical encoding...")
    cyclical_cols = [col for col in X_train.columns if '_sin' in col or '_cos' in col]
    if len(cyclical_cols) > 0:
        print(f"   ✓ Cyclical encoding applied: {len(cyclical_cols)} features")
        print(f"     Cyclical features: {cyclical_cols}")
    else:
        print("   ✗ No cyclical features found")
    
    # Check 5: All ESI levels present
    print("\n5. Checking ESI level distribution...")
    all_esi = sorted(set(y_train) | set(y_val) | set(y_test))
    print(f"   ESI levels present: {[int(x) for x in all_esi]}")
    
    # Check 6: Feature count reduction
    print("\n6. Feature count analysis...")
    original_features = len(df.columns) - 2  # Exclude target and year
    final_features = len(X_train.columns)
    print(f"   Original features: {original_features}")
    print(f"   Final features: {final_features}")
    print(f"   Reduction: {original_features - final_features} features")
    
    # Check 7: Data types
    print("\n7. Checking data types...")
    numeric_only = all(X_train[col].dtype in ['float64', 'int64'] for col in X_train.columns)
    if numeric_only:
        print("   ✓ All features are numeric (ready for ML)")
    else:
        print("   ✗ Non-numeric features found")
        print(f"     Types: {X_train.dtypes.value_counts()}")
    
    print("\n" + "=" * 60)
    print("PREPROCESSING TEST COMPLETE")
    print("=" * 60)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = test_preprocessing_pipeline()
    
    print("\nSample of preprocessed training data:")
    print(X_train.head())
    print(f"\nTarget distribution (train):")
    print(y_train.value_counts().sort_index())

