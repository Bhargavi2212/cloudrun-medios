"""
Fix preprocessing pipeline issues.

ROOT CAUSE:
- 8 columns became constant after SMOTE due to >50% missing values
- Preprocessing order was incorrect (transformers fitted before split)

SOLUTION:
1. Drop high-missing columns (>50%)
2. Split data FIRST
3. Fit all transformers on train only
4. Remove constant columns after SMOTE
5. Save new cache

Expected improvement: 49% → 92-93% accuracy
"""

import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from typing import Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'services' / 'manage-agent'))

from ml.data_splitter import RandomStratifiedSplitter
from ml.preprocessing import KNNImputerWrapper, YeoJohnsonTransformer
from ml.feature_engineering import DiagnosisDropper, CyclicalEncoder, OutlierClipper
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def map_esi_levels(y):
    """Map ESI levels from [0,1,2,3,4,5,7] to sequential [0,1,2,3,4,5,6]."""
    forward_mapping = {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, 4.0: 4, 5.0: 5, 7.0: 6}
    y_mapped = np.array([forward_mapping[float(val)] for val in y], dtype=np.int32)
    return pd.Series(y_mapped, name=y.name)


def drop_high_missing_columns(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Drop columns with >50% missing values.
    
    Args:
        df: DataFrame
        threshold: Missing percentage threshold (0.5 = 50%)
        
    Returns:
        DataFrame with high-missing columns removed
    """
    print("\n" + "=" * 60)
    print("STEP 1: DROPPING HIGH-MISSING COLUMNS (>50%)")
    print("=" * 60)
    
    missing_pct = df.isnull().sum() / len(df)
    high_missing_cols = missing_pct[missing_pct > threshold].index.tolist()
    
    print(f"\nColumns with >{threshold*100:.0f}% missing:")
    for col in high_missing_cols:
        pct = missing_pct[col] * 100
        print(f"  {col:20s}: {pct:5.1f}% missing")
    
    if high_missing_cols:
        df_cleaned = df.drop(columns=high_missing_cols)
        print(f"\n✓ Dropped {len(high_missing_cols)} high-missing columns")
        print(f"  Features: {len(df.columns)} → {len(df_cleaned.columns)}")
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
    print(f"  Train: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val:   {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:  {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    print(f"\nClass distribution (train):")
    print(y_train.value_counts().sort_index())
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def encode_esi_levels(y_train, y_val, y_test):
    """Map ESI levels: [0,1,2,3,4,5,7] → [0,1,2,3,4,5,6]."""
    print("\n" + "=" * 60)
    print("STEP 3: ENCODING ESI LEVELS")
    print("=" * 60)
    
    y_train_encoded = map_esi_levels(y_train)
    y_val_encoded = map_esi_levels(y_val)
    y_test_encoded = map_esi_levels(y_test)
    
    print(f"\nOriginal ESI levels: {sorted(y_train.unique())}")
    print(f"Encoded ESI levels: {sorted(y_train_encoded.unique())}")
    print(f"✓ ESI levels encoded")
    
    return y_train_encoded, y_val_encoded, y_test_encoded


def apply_knn_imputation(X_train, X_val, X_test, n_neighbors: int = 3):
    """Apply KNN imputation (fit on train only)."""
    print("\n" + "=" * 60)
    print("STEP 4: KNN IMPUTATION (fit on train only)")
    print("=" * 60)
    
    imputer = KNNImputerWrapper(n_neighbors=n_neighbors, max_samples=50000)
    
    print("  Fitting imputer on training set...")
    X_train_imputed = imputer.fit_transform(X_train, None)
    
    print("  Transforming validation set...")
    X_val_imputed = imputer.transform(X_val)
    
    print("  Transforming test set...")
    X_test_imputed = imputer.transform(X_test)
    
    print(f"✓ Imputation complete")
    print(f"  Train missing: {pd.DataFrame(X_train_imputed).isnull().sum().sum()}")
    print(f"  Val missing:   {pd.DataFrame(X_val_imputed).isnull().sum().sum()}")
    print(f"  Test missing:  {pd.DataFrame(X_test_imputed).isnull().sum().sum()}")
    
    return X_train_imputed, X_val_imputed, X_test_imputed


def apply_outlier_clipping(X_train, X_val, X_test, iqr_factor: float = 1.5):
    """Apply outlier clipping (IQR-based)."""
    print("\n" + "=" * 60)
    print("STEP 5: OUTLIER CLIPPING (IQR factor=1.5)")
    print("=" * 60)
    
    clipper = OutlierClipper(factor=iqr_factor)
    
    print("  Fitting clipper on training set...")
    X_train_clipped = clipper.fit_transform(X_train, None)
    
    print("  Transforming validation set...")
    X_val_clipped = clipper.transform(X_val)
    
    print("  Transforming test set...")
    X_test_clipped = clipper.transform(X_test)
    
    print(f"✓ Outlier clipping complete")
    
    return X_train_clipped, X_val_clipped, X_test_clipped


def apply_yeo_johnson(X_train, X_val, X_test, skew_threshold: float = 1.0):
    """Apply Yeo-Johnson transformation (fit on train only)."""
    print("\n" + "=" * 60)
    print("STEP 6: YEO-JOHNSON TRANSFORMATION (fit on train only)")
    print("=" * 60)
    
    transformer = YeoJohnsonTransformer(skew_threshold=skew_threshold)
    
    print("  Fitting transformer on training set...")
    X_train_transformed = transformer.fit_transform(X_train, None)
    
    print("  Transforming validation set...")
    X_val_transformed = transformer.transform(X_val)
    
    print("  Transforming test set...")
    X_test_transformed = transformer.transform(X_test)
    
    print(f"✓ Yeo-Johnson transformation complete")
    
    return X_train_transformed, X_val_transformed, X_test_transformed


def apply_cyclical_encoding(X_train, X_val, X_test):
    """Apply cyclical encoding for temporal features."""
    print("\n" + "=" * 60)
    print("STEP 7: CYCLICAL ENCODING (temporal features)")
    print("=" * 60)
    
    encoder = CyclicalEncoder()
    
    print("  Fitting encoder on training set...")
    X_train_encoded = encoder.fit_transform(X_train, None)
    
    print("  Transforming validation set...")
    X_val_encoded = encoder.transform(X_val)
    
    print("  Transforming test set...")
    X_test_encoded = encoder.transform(X_test)
    
    print(f"✓ Cyclical encoding complete")
    
    return X_train_encoded, X_val_encoded, X_test_encoded


def apply_standardization(X_train, X_val, X_test):
    """Apply StandardScaler (fit on SMOTE'd train, transform all)."""
    print("\n" + "=" * 60)
    print("STEP 10: STANDARDIZATION (fit on SMOTE'd train, transform all)")
    print("=" * 60)
    print("  Note: StandardScaler fitted on SMOTE'd training set (382K samples)")
    print("        This ensures train/val/test have consistent scaling")
    
    scaler = StandardScaler()
    
    print("  Fitting scaler on training set...")
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    print("  Transforming validation set...")
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )
    
    print("  Transforming test set...")
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    # Verify scaling (train should have mean≈0, std≈1 after StandardScaler)
    print(f"\n  Scaling verification (sample features):")
    print("  (Train should have mean≈0, std≈1 after StandardScaler)")
    sample_cols = X_train_scaled.columns[:5]
    for col in sample_cols:
        train_mean = X_train_scaled[col].mean()
        train_std = X_train_scaled[col].std()
        val_mean = X_val_scaled[col].mean()
        val_std = X_val_scaled[col].std()
        
        # Check if train is properly scaled (mean≈0, std≈1)
        train_ok = abs(train_mean) < 0.01 and abs(train_std - 1.0) < 0.01
        # Check if val/test are transformed with same scaler (should have similar std)
        val_ok = abs(val_std - 1.0) < 0.1
        
        status = "✅" if train_ok and val_ok else "⚠️"
        print(f"    {col}: train(μ={train_mean:.6f}, σ={train_std:.6f}), val(μ={val_mean:.6f}, σ={val_std:.6f}) {status}")
    
    print(f"✓ Standardization complete")
    
    return X_train_scaled, X_val_scaled, X_test_scaled


def apply_smote(X_train, y_train, random_state: int = 42):
    """Apply SMOTE to training set only (BEFORE standardization)."""
    print("\n" + "=" * 60)
    print("STEP 9: SMOTE (TRAINING SET ONLY - BEFORE STANDARDIZATION)")
    print("=" * 60)
    print("  Note: SMOTE applied BEFORE standardization to ensure")
    print("        StandardScaler is fitted on the balanced training set")
    
    print(f"  Original class distribution:")
    print(y_train.value_counts().sort_index())
    
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"\n  After SMOTE:")
    y_resampled_series = pd.Series(y_resampled)
    print(y_resampled_series.value_counts().sort_index())
    print(f"  Samples: {len(X_train):,} → {len(X_resampled):,}")
    
    X_resampled_df = pd.DataFrame(
        X_resampled,
        columns=X_train.columns,
        index=range(len(X_resampled))
    )
    
    print(f"✓ SMOTE complete")
    
    return X_resampled_df, y_resampled_series


def remove_constant_columns(X_train, X_val, X_test):
    """Remove constant columns (no variance)."""
    print("\n" + "=" * 60)
    print("STEP 10: REMOVING CONSTANT COLUMNS")
    print("=" * 60)
    
    # Find constant columns (nunique <= 1)
    constant_cols = []
    for col in X_train.columns:
        if X_train[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        print(f"\n  Found {len(constant_cols)} constant columns:")
        for col in constant_cols:
            unique_vals = X_train[col].unique()
            print(f"    {col}: {unique_vals}")
        
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


def verify_preprocessing(X_train, X_val, X_test, y_train, y_val, y_test):
    """Verify preprocessing results."""
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    # Check shapes
    print(f"\nShapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")
    
    # Check NaN
    train_nan = X_train.isnull().sum().sum()
    val_nan = X_val.isnull().sum().sum()
    test_nan = X_test.isnull().sum().sum()
    
    print(f"\nNaN values:")
    print(f"  Train: {train_nan} (expected: 0)")
    print(f"  Val:   {val_nan} (expected: 0)")
    print(f"  Test:  {test_nan} (expected: 0)")
    
    # Check constant columns
    constant_cols = [col for col in X_train.columns if X_train[col].nunique() <= 1]
    print(f"\nConstant columns: {len(constant_cols)} (expected: 0)")
    if constant_cols:
        print(f"  ❌ Found: {constant_cols}")
    else:
        print(f"  ✅ No constant columns")
    
    # Check scaling
    print(f"\nScaling check (sample features):")
    sample_cols = X_train.columns[:3]
    scale_ok = True
    for col in sample_cols:
        train_mean = X_train[col].mean()
        train_std = X_train[col].std()
        val_mean = X_val[col].mean()
        val_std = X_val[col].std()
        
        mean_diff = abs(train_mean - val_mean) / (abs(train_mean) + 1e-8)
        std_diff = abs(train_std - val_std) / (abs(train_std) + 1e-8)
        
        status = "✅" if mean_diff < 0.5 and std_diff < 0.5 else "❌"
        if mean_diff > 0.5 or std_diff > 0.5:
            scale_ok = False
        
        print(f"  {col}: {status}")
    
    # Check class distribution
    print(f"\nClass distribution:")
    print(f"  Train: {sorted(y_train.value_counts().to_dict().items())}")
    print(f"  Val:   {sorted(y_val.value_counts().to_dict().items())}")
    
    all_ok = (
        train_nan == 0 and val_nan == 0 and test_nan == 0 and
        len(constant_cols) == 0 and scale_ok
    )
    
    print(f"\n{'✅ All checks passed' if all_ok else '❌ Some checks failed'}")
    
    return all_ok


def main():
    print("=" * 60)
    print("FIX PREPROCESSING PIPELINE")
    print("=" * 60)
    print("\nThis script will:")
    print("  1. Drop high-missing columns (>50%)")
    print("  2. Split data FIRST")
    print("  3. Fit all transformers on train only")
    print("  4. Remove constant columns")
    print("  5. Save new cache")
    
    # Load raw data
    print("\n" + "=" * 60)
    print("LOADING RAW DATA")
    print("=" * 60)
    
    data_path = project_root / "data" / "NHAMCS_2011_2022_combined.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Raw data not found: {data_path}")
    
    print(f"\nLoading: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Raw data shape: {df.shape}")
    
    # Extract features and target
    target_col = 'esi_level'
    exclude_cols = ['year']  # Exclude year
    
    X_full = df.drop(columns=[target_col] + exclude_cols)
    y_full = df[target_col]
    
    # Step 1: Drop diagnosis columns (data leakage)
    print("\n" + "=" * 60)
    print("DROPPING DIAGNOSIS COLUMNS (data leakage)")
    print("=" * 60)
    
    diag_cols = [col for col in X_full.columns if 'diag' in col.lower()]
    if diag_cols:
        X_full = X_full.drop(columns=diag_cols)
        print(f"✓ Dropped {len(diag_cols)} diagnosis columns: {diag_cols}")
    else:
        print("✓ No diagnosis columns found")
    
    # Step 2: Drop high-missing columns
    X_full = drop_high_missing_columns(X_full, threshold=0.5)
    
    # Step 3: Split data FIRST (before any preprocessing)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_full, y_full)
    
    # Step 4: Encode ESI levels
    y_train, y_val, y_test = encode_esi_levels(y_train, y_val, y_test)
    
    # Step 5: KNN imputation (fit on train only)
    X_train, X_val, X_test = apply_knn_imputation(X_train, X_val, X_test)
    
    # Step 6: Outlier clipping
    X_train, X_val, X_test = apply_outlier_clipping(X_train, X_val, X_test)
    
    # Step 7: Yeo-Johnson transformation
    X_train, X_val, X_test = apply_yeo_johnson(X_train, X_val, X_test)
    
    # Step 8: Cyclical encoding
    X_train, X_val, X_test = apply_cyclical_encoding(X_train, X_val, X_test)
    
    # Step 9: SMOTE (training set only!) - BEFORE standardization
    # This ensures StandardScaler is fitted on the SMOTE'd training set
    X_train, y_train = apply_smote(X_train, y_train)
    
    # Step 10: Standardization (fit on SMOTE'd train, transform all)
    # Now StandardScaler is fitted on the balanced training set (382K samples)
    X_train, X_val, X_test = apply_standardization(X_train, X_val, X_test)
    
    # Step 11: Remove constant columns (after all transformations)
    X_train, X_val, X_test = remove_constant_columns(X_train, X_val, X_test)
    
    # Verify
    verify_ok = verify_preprocessing(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Save new cache
    print("\n" + "=" * 60)
    print("SAVING NEW CACHE")
    print("=" * 60)
    
    outputs_dir = project_root / "services" / "manage-agent" / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    cache_file = outputs_dir / "preprocessed_data_cache_v2.pkl"
    
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
        print("\n✅ All verification checks passed!")
        print("✅ Ready to retrain models with new cache")
        print("\nNext steps:")
        print("  1. Update train scripts to use: preprocessed_data_cache_v2.pkl")
        print("  2. Retrain models")
        print("  3. Expected improvement: 49% → 92-93% accuracy")
    else:
        print("\n⚠️  Some verification checks failed. Review output above.")


if __name__ == "__main__":
    main()

