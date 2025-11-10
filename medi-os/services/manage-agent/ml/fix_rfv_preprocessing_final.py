"""
Fix RFV preprocessing - separate numeric from categorical.

CRITICAL FIX: StandardScaler was applied to RFV codes, destroying their meaning.
Solution: Only scale numeric features, keep RFV in original range.
"""

import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from typing import Tuple, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'services' / 'manage-agent'))

from ml.data_splitter import RandomStratifiedSplitter
from ml.preprocessing import KNNImputerWrapper, YeoJohnsonTransformer
from ml.feature_engineering import DiagnosisDropper, CyclicalEncoder, OutlierClipper
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTENC


def get_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Separate features into numeric, categorical, and binary.
    
    Returns:
        numeric_cols: Features to scale (age, vitals)
        categorical_cols: One-hot RFV features (don't scale - already binary)
        binary_cols: Binary features (don't scale)
    """
    # One-hot RFV fields (already binary, don't scale)
    rfv_onehot_fields = [f for f in X.columns if f.startswith('rfv1_') or f.startswith('rfv2_')]
    
    # RFV 3D fields (numeric, but don't scale - keep original range)
    rfv_3d_fields = ['rfv1_3d', 'rfv2_3d']
    rfv_3d_fields = [f for f in rfv_3d_fields if f in X.columns]
    
    # Combine RFV fields (one-hot + 3D)
    rfv_fields = rfv_onehot_fields + rfv_3d_fields
    
    # Binary features (already 0/1)
    binary_fields = ['sex', 'injury', 'ambulance_arrival', 'seen_72h', 
                     'discharged_7d', 'cebvd', 'chf', 'ed_dialysis', 'hiv', 
                     'diabetes', 'no_chronic_conditions', 'on_oxygen']
    binary_fields = [f for f in binary_fields if f in X.columns]
    
    # Temporal (cyclical encoded - don't scale again)
    temporal_fields = ['month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos']
    temporal_fields = [f for f in temporal_fields if f in X.columns]
    
    # Everything else is numeric (age, vitals, etc.)
    all_cols = set(X.columns)
    categorical_cols = rfv_fields
    binary_cols = [f for f in binary_fields if f in all_cols]
    temporal_cols = [f for f in temporal_fields if f in all_cols]
    
    # Numeric = everything except RFV, binary, temporal
    excluded = set(categorical_cols + binary_cols + temporal_cols)
    numeric_cols = [f for f in all_cols if f not in excluded]
    
    return numeric_cols, categorical_cols, binary_cols


def map_esi_levels(y):
    """Map ESI levels: [0,1,2,3,4,5,7] → [0,1,2,3,4,5,6]."""
    forward_mapping = {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, 4.0: 4, 5.0: 5, 7.0: 6}
    y_mapped = np.array([forward_mapping[float(val)] for val in y], dtype=np.int32)
    return pd.Series(y_mapped, name=y.name)


def drop_high_missing_columns(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Drop columns with >50% missing values."""
    print("\n" + "=" * 60)
    print("STEP 1: DROPPING HIGH-MISSING COLUMNS (>50%)")
    print("=" * 60)
    
    missing_pct = df.isnull().sum() / len(df)
    high_missing_cols = missing_pct[missing_pct > threshold].index.tolist()
    
    if high_missing_cols:
        print(f"\nDropping {len(high_missing_cols)} high-missing columns:")
        for col in high_missing_cols:
            pct = missing_pct[col] * 100
            print(f"  {col:20s}: {pct:5.1f}% missing")
        df_cleaned = df.drop(columns=high_missing_cols)
        print(f"\n✓ Features: {len(df.columns)} → {len(df_cleaned.columns)}")
    else:
        df_cleaned = df
        print("\n✓ No high-missing columns to drop")
    
    return df_cleaned


def split_data(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> Tuple:
    """Split data FIRST before any preprocessing."""
    print("\n" + "=" * 60)
    print("STEP 2: SPLITTING DATA (70/15/15) - FIRST STEP")
    print("=" * 60)
    
    splitter = RandomStratifiedSplitter(
        train_size=0.70,
        val_size=0.15,
        random_state=random_state
    )
    
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(X, y)
    
    print(f"\nData split completed:")
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Val:   {len(X_val):,} samples")
    print(f"  Test:  {len(X_test):,} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def encode_esi_levels(y_train, y_val, y_test):
    """Map ESI levels: [0,1,2,3,4,5,7] → [0,1,2,3,4,5,6]."""
    print("\n" + "=" * 60)
    print("STEP 3: ENCODING ESI LEVELS")
    print("=" * 60)
    
    y_train_encoded = map_esi_levels(y_train)
    y_val_encoded = map_esi_levels(y_val)
    y_test_encoded = map_esi_levels(y_test)
    
    print(f"✓ ESI levels encoded: {sorted(y_train.unique())} → {sorted(y_train_encoded.unique())}")
    
    return y_train_encoded, y_val_encoded, y_test_encoded


def apply_knn_imputation(X_train, X_val, X_test, numeric_cols: List[str]):
    """Apply KNN imputation to ALL features (including RFV)."""
    print("\n" + "=" * 60)
    print("STEP 4: KNN IMPUTATION (fit on train only)")
    print("=" * 60)
    
    imputer = KNNImputerWrapper(n_neighbors=3, max_samples=50000)
    
    print("  Fitting imputer on training set...")
    X_train_imputed = imputer.fit_transform(X_train, None)
    
    print("  Transforming validation set...")
    X_val_imputed = imputer.transform(X_val)
    
    print("  Transforming test set...")
    X_test_imputed = imputer.transform(X_test)
    
    print(f"✓ Imputation complete")
    
    return X_train_imputed, X_val_imputed, X_test_imputed


def apply_outlier_clipping(X_train, X_val, X_test, numeric_cols: List[str]):
    """Apply outlier clipping ONLY to numeric features (not RFV)."""
    print("\n" + "=" * 60)
    print("STEP 5: OUTLIER CLIPPING (numeric features only)")
    print("=" * 60)
    
    clipper = OutlierClipper(factor=1.5)
    
    # Fit on train (numeric columns only)
    X_train_numeric = X_train[numeric_cols]
    clipper.fit(X_train_numeric, None)
    
    # Transform all sets
    X_train_clipped = X_train.copy()
    X_train_clipped[numeric_cols] = clipper.transform(X_train_numeric)
    
    X_val_clipped = X_val.copy()
    X_val_clipped[numeric_cols] = clipper.transform(X_val[numeric_cols])
    
    X_test_clipped = X_test.copy()
    X_test_clipped[numeric_cols] = clipper.transform(X_test[numeric_cols])
    
    print(f"✓ Outlier clipping complete (applied to {len(numeric_cols)} numeric features)")
    
    return X_train_clipped, X_val_clipped, X_test_clipped


def apply_yeo_johnson(X_train, X_val, X_test, numeric_cols: List[str]):
    """Apply Yeo-Johnson transformation ONLY to numeric features."""
    print("\n" + "=" * 60)
    print("STEP 6: YEO-JOHNSON TRANSFORMATION (numeric features only)")
    print("=" * 60)
    
    transformer = YeoJohnsonTransformer(skew_threshold=1.0)
    
    # Fit on train (numeric columns only)
    X_train_numeric = X_train[numeric_cols]
    X_train_transformed_numeric = transformer.fit_transform(X_train_numeric, None)
    
    # Transform all sets
    X_train_transformed = X_train.copy()
    X_train_transformed[numeric_cols] = X_train_transformed_numeric
    
    X_val_transformed = X_val.copy()
    X_val_transformed[numeric_cols] = transformer.transform(X_val[numeric_cols])
    
    X_test_transformed = X_test.copy()
    X_test_transformed[numeric_cols] = transformer.transform(X_test[numeric_cols])
    
    print(f"✓ Yeo-Johnson transformation complete")
    
    return X_train_transformed, X_val_transformed, X_test_transformed


def apply_cyclical_encoding(X_train, X_val, X_test):
    """Apply cyclical encoding for temporal features."""
    print("\n" + "=" * 60)
    print("STEP 7: CYCLICAL ENCODING (temporal features)")
    print("=" * 60)
    
    encoder = CyclicalEncoder()
    
    X_train_encoded = encoder.fit_transform(X_train, None)
    X_val_encoded = encoder.transform(X_val)
    X_test_encoded = encoder.transform(X_test)
    
    print(f"✓ Cyclical encoding complete")
    
    return X_train_encoded, X_val_encoded, X_test_encoded


def apply_smote_with_rfv(X_train, y_train, categorical_cols: List[str], random_state: int = 42):
    """Apply SMOTENC (SMOTE for Nominal and Continuous) with RFV as categorical."""
    print("\n" + "=" * 60)
    print("STEP 8: SMOTE-NC (TRAINING SET ONLY - BEFORE SCALING)")
    print("=" * 60)
    print("  Note: RFV one-hot features treated as categorical (not scaled)")
    
    print(f"\n  Original class distribution:")
    print(y_train.value_counts().sort_index())
    
    # Get categorical feature indices (RFV one-hot columns, exclude 3D numeric)
    categorical_indices = []
    for i, col in enumerate(X_train.columns):
        if col in categorical_cols and not col.endswith('_3d'):
            categorical_indices.append(i)
    
    if categorical_indices:
        print(f"\n  SMOTE-NC: {len(categorical_indices)} categorical features (RFV one-hot)")
        smote = SMOTENC(
            categorical_features=categorical_indices,
            random_state=random_state
        )
    else:
        print(f"\n  Warning: No categorical features detected, using standard SMOTE")
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=random_state)
    
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"\n  After SMOTE-NC:")
    y_resampled_series = pd.Series(y_resampled)
    print(y_resampled_series.value_counts().sort_index())
    print(f"  Samples: {len(X_train):,} → {len(X_resampled):,}")
    
    X_resampled_df = pd.DataFrame(
        X_resampled,
        columns=X_train.columns,
        index=range(len(X_resampled))
    )
    
    print(f"✓ SMOTE-NC complete")
    
    return X_resampled_df, y_resampled_series


def apply_selective_standardization(X_train, X_val, X_test, numeric_cols: List[str]):
    """Apply StandardScaler ONLY to numeric features (NOT RFV)."""
    print("\n" + "=" * 60)
    print("STEP 9: SELECTIVE STANDARDIZATION (numeric features only)")
    print("=" * 60)
    print("  CRITICAL: RFV codes NOT scaled (kept in original range)")
    print(f"  Scaling {len(numeric_cols)} numeric features")
    print(f"  Keeping RFV codes in original range")
    
    scaler = StandardScaler()
    
    # Fit scaler on numeric features only
    X_train_numeric = X_train[numeric_cols]
    X_train_scaled_numeric = scaler.fit_transform(X_train_numeric)
    
    # Create scaled DataFrames
    X_train_scaled = X_train.copy()
    X_train_scaled[numeric_cols] = X_train_scaled_numeric
    
    X_val_scaled = X_val.copy()
    X_val_scaled[numeric_cols] = scaler.transform(X_val[numeric_cols])
    
    X_test_scaled = X_test.copy()
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    # Verify RFV NOT scaled
    print(f"\n  Verification:")
    rfv_fields = [f for f in X_train.columns if 'rfv' in f.lower()]
    for rfv in rfv_fields[:3]:  # Check first 3 RFV fields
        train_min = X_train_scaled[rfv].min()
        train_max = X_train_scaled[rfv].max()
        train_mean = X_train_scaled[rfv].mean()
        train_std = X_train_scaled[rfv].std()
        
        # Check if NOT scaled (mean≈0, std≈1, range in [-3, 3] means scaled)
        is_scaled = abs(train_mean) < 0.5 and abs(train_std - 1.0) < 0.5 and abs(train_min) < 3 and abs(train_max) < 3
        is_original_range = not is_scaled and (train_max > 100 or abs(train_mean) > 10)
        
        status = "✅ Original range" if is_original_range else ("❌ SCALED (WRONG!)" if is_scaled else "⚠️ Unknown")
        print(f"    {rfv}: range=[{train_min:.0f}, {train_max:.0f}], mean={train_mean:.0f} {status}")
    
    # Check numeric features ARE scaled
    for num_col in numeric_cols[:3]:  # Check first 3 numeric
        train_mean = X_train_scaled[num_col].mean()
        train_std = X_train_scaled[num_col].std()
        is_scaled = abs(train_mean) < 0.1 and abs(train_std - 1.0) < 0.1
        status = "✅ Scaled" if is_scaled else "⚠️"
        print(f"    {num_col}: mean={train_mean:.4f}, std={train_std:.4f} {status}")
    
    print(f"\n✓ Selective standardization complete")
    
    return X_train_scaled, X_val_scaled, X_test_scaled


def remove_constant_columns(X_train, X_val, X_test):
    """Remove constant columns (no variance)."""
    print("\n" + "=" * 60)
    print("STEP 10: REMOVING CONSTANT COLUMNS")
    print("=" * 60)
    
    constant_cols = []
    for col in X_train.columns:
        if X_train[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        print(f"\n  Found {len(constant_cols)} constant columns:")
        for col in constant_cols:
            print(f"    {col}")
        
        X_train_cleaned = X_train.drop(columns=constant_cols)
        X_val_cleaned = X_val.drop(columns=constant_cols)
        X_test_cleaned = X_test.drop(columns=constant_cols)
        
        print(f"\n✓ Removed {len(constant_cols)} constant columns")
        print(f"  Features: {len(X_train.columns)} → {len(X_train_cleaned.columns)}")
    else:
        print(f"\n✓ No constant columns found")
        X_train_cleaned = X_train
        X_val_cleaned = X_val
        X_test_cleaned = X_test
    
    return X_train_cleaned, X_val_cleaned, X_test_cleaned


def verify_rfv_ranges(X_train, categorical_cols: List[str]):
    """Verify RFV features are correctly handled."""
    print("\n" + "=" * 60)
    print("RFV FEATURE VERIFICATION")
    print("=" * 60)
    
    print("\nRFV one-hot features should be binary (0/1)")
    print("RFV 3D features should be in original range (0-8999)")
    
    all_ok = True
    
    # Check one-hot RFV features
    rfv_onehot = [c for c in categorical_cols if (c.startswith('rfv1_') or c.startswith('rfv2_')) and not c.endswith('_3d')]
    if rfv_onehot:
        print(f"\nRFV one-hot features ({len(rfv_onehot)}):")
        sample_onehot = rfv_onehot[:5]
        for rfv in sample_onehot:
            if rfv in X_train.columns:
                values = X_train[rfv]
                min_val = values.min()
                max_val = values.max()
                unique_vals = values.unique()
                is_binary = set(unique_vals).issubset({0, 1})
                status = "✅ Binary" if is_binary else "❌ NOT binary"
                print(f"  {rfv}: range=[{min_val:.0f}, {max_val:.0f}], unique={unique_vals} {status}")
                if not is_binary:
                    all_ok = False
    
    # Check RFV 3D features
    rfv_3d = [c for c in categorical_cols if c.endswith('_3d')]
    if rfv_3d:
        print(f"\nRFV 3D features ({len(rfv_3d)}):")
        for rfv in rfv_3d:
            if rfv in X_train.columns:
                values = X_train[rfv]
                min_val = values.min()
                max_val = values.max()
                mean_val = values.mean()
                is_original = max_val > 100 and max_val < 10000
                status = "✅ Original range" if is_original else "❌ SCALED"
                print(f"  {rfv}: range=[{min_val:.0f}, {max_val:.0f}], mean={mean_val:.0f} {status}")
                if not is_original:
                    all_ok = False
    
    if all_ok:
        print(f"\n✅ All RFV features correctly handled")
    else:
        print(f"\n❌ Some RFV features have issues")
    
    return all_ok


def main():
    print("=" * 60)
    print("FIX RFV PREPROCESSING - SELECTIVE SCALING")
    print("=" * 60)
    print("\nThis script will:")
    print("  1. Use RFV-fixed CSV (one-hot encoded RFV1+RFV2, dropped RFV3)")
    print("  2. Separate features: numeric vs RFV (one-hot + 3D)")
    print("  3. Scale ONLY numeric features (age, vitals)")
    print("  4. Keep RFV one-hot as binary, RFV 3D in original range")
    print("  5. Use SMOTENC for proper categorical handling")
    
    # Load raw data
    print("\n" + "=" * 60)
    print("LOADING RAW DATA")
    print("=" * 60)
    
    # Try RFV-fixed CSV first, fallback to original
    data_path_fixed = project_root / "data" / "NHAMCS_2011_2022_combined_rfv_fixed.csv"
    data_path_original = project_root / "data" / "NHAMCS_2011_2022_combined.csv"
    
    if data_path_fixed.exists():
        data_path = data_path_fixed
        print(f"\nUsing RFV-fixed CSV: {data_path}")
    elif data_path_original.exists():
        data_path = data_path_original
        print(f"\nUsing original CSV: {data_path}")
    else:
        raise FileNotFoundError(f"Raw data not found")
    
    print(f"\nLoading: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Raw data shape: {df.shape}")
    
    # Extract features and target
    target_col = 'esi_level'
    exclude_cols = ['year']
    
    X_full = df.drop(columns=[target_col] + exclude_cols)
    y_full = df[target_col]
    
    # Step 1: Drop diagnosis columns
    print("\n" + "=" * 60)
    print("DROPPING DIAGNOSIS COLUMNS (data leakage)")
    print("=" * 60)
    
    diag_cols = [col for col in X_full.columns if 'diag' in col.lower()]
    if diag_cols:
        X_full = X_full.drop(columns=diag_cols)
        print(f"✓ Dropped {len(diag_cols)} diagnosis columns: {diag_cols}")
    
    # Step 2: Drop high-missing columns
    X_full = drop_high_missing_columns(X_full, threshold=0.5)
    
    # Step 3: Split data FIRST
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_full, y_full)
    
    # Step 4: Encode ESI levels
    y_train, y_val, y_test = encode_esi_levels(y_train, y_val, y_test)
    
    # Step 5: KNN imputation (all features - before other transformations)
    X_train, X_val, X_test = apply_knn_imputation(X_train, X_val, X_test, [])
    
    # Step 6: Identify feature types (before transformations)
    print("\n" + "=" * 60)
    print("IDENTIFYING FEATURE TYPES")
    print("=" * 60)
    
    numeric_cols, categorical_cols, binary_cols = get_feature_types(X_train)
    
    print(f"\nFeature types:")
    print(f"  Numeric (to scale):     {len(numeric_cols)} features")
    print(f"    {numeric_cols[:5]}{'...' if len(numeric_cols) > 5 else ''}")
    print(f"  RFV (one-hot + 3D):     {len(categorical_cols)} features")
    print(f"    One-hot RFV: {len([c for c in categorical_cols if c.startswith('rfv1_') or c.startswith('rfv2_')])}")
    print(f"    RFV 3D: {len([c for c in categorical_cols if '_3d' in c])}")
    print(f"  Binary (don't scale):  {len(binary_cols)} features")
    print(f"    {binary_cols[:5]}{'...' if len(binary_cols) > 5 else ''}")
    
    # Step 7: Outlier clipping (numeric only)
    X_train, X_val, X_test = apply_outlier_clipping(X_train, X_val, X_test, numeric_cols)
    
    # Step 8: Yeo-Johnson (numeric only)
    X_train, X_val, X_test = apply_yeo_johnson(X_train, X_val, X_test, numeric_cols)
    
    # Step 9: Cyclical encoding (replaces month/day_of_week with sin/cos)
    X_train, X_val, X_test = apply_cyclical_encoding(X_train, X_val, X_test)
    
    # Re-identify feature types after cyclical encoding
    numeric_cols, categorical_cols, binary_cols = get_feature_types(X_train)
    print(f"\nFeature types (after cyclical encoding):")
    print(f"  Numeric (to scale):     {len(numeric_cols)} features")
    print(f"  Categorical (RFV):      {len(categorical_cols)} features")
    print(f"  Binary (don't scale):  {len(binary_cols)} features")
    
    # Step 10: SMOTE-NC (before scaling, with RFV as categorical)
    X_train, y_train = apply_smote_with_rfv(X_train, y_train, categorical_cols)
    
    # Step 11: Selective standardization (numeric only, exclude RFV)
    X_train, X_val, X_test = apply_selective_standardization(X_train, X_val, X_test, numeric_cols)
    
    # Step 12: Remove constant columns
    X_train, X_val, X_test = remove_constant_columns(X_train, X_val, X_test)
    
    # Verify RFV features
    verify_ok = verify_rfv_ranges(X_train, categorical_cols)
    
    # Save new cache
    print("\n" + "=" * 60)
    print("SAVING NEW CACHE")
    print("=" * 60)
    
    outputs_dir = project_root / "services" / "manage-agent" / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    cache_file = outputs_dir / "preprocessed_data_cache_v4.pkl"
    
    preprocessed_data = {
        'train': X_train,
        'val': X_val,
        'test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }
    
    print(f"\nSaving to: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(preprocessed_data, f)
    
    file_size = cache_file.stat().st_size / (1024 * 1024)  # MB
    print(f"✓ Cache saved! Size: {file_size:.2f} MB")
    print(f"  Train: {len(X_train):,} x {len(X_train.columns)}")
    print(f"  Val:   {len(X_val):,} x {len(X_val.columns)}")
    print(f"  Test:  {len(X_test):,} x {len(X_test.columns)}")
    
    print("\n" + "=" * 60)
    print("PREPROCESSING FIX COMPLETE")
    print("=" * 60)
    
    if verify_ok:
        print("\n✅ RFV codes verified in original range!")
        print("✅ Ready to retrain models")
        print("\nExpected improvement:")
        print("  Before: 50% accuracy, RFV importance 0.047")
        print("  After:  85-92% accuracy, RFV importance >0.25")
    else:
        print("\n⚠️  RFV verification failed. Review output above.")


if __name__ == "__main__":
    main()

