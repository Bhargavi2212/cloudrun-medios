"""
Comprehensive Data Quality Diagnostic

Identifies root cause of poor model performance (21-53% accuracy vs expected 85-92%).
"""

import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'services' / 'manage-agent'))


def print_section(title, char="="):
    """Print a formatted section header."""
    print("\n" + char * 80)
    print(f"{title}")
    print(char * 80)


def main():
    print_section("COMPREHENSIVE DATA QUALITY DIAGNOSTIC", "=")
    
    # ============================================================================
    # 1. ORIGINAL DATA VERIFICATION
    # ============================================================================
    print_section("1. ORIGINAL DATA VERIFICATION")
    
    raw_data_path = project_root / "data" / "NHAMCS_2011_2022_combined.csv"
    
    if not raw_data_path.exists():
        print(f"❌ Raw data not found: {raw_data_path}")
        return
    
    print(f"\nLoading raw data: {raw_data_path}")
    df_raw = pd.read_csv(raw_data_path)
    
    print(f"\nRaw data shape: {df_raw.shape}")
    print(f"Columns: {list(df_raw.columns)}")
    
    # Check years
    if 'year' in df_raw.columns:
        print(f"\nYear distribution:")
        print(df_raw['year'].value_counts().sort_index())
    else:
        print("\n⚠️  'year' column not found")
    
    # Check ESI distribution in original data
    if 'esi_level' in df_raw.columns:
        print(f"\nESI distribution in ORIGINAL data:")
        esi_counts = df_raw['esi_level'].value_counts().sort_index()
        print(esi_counts)
        print(f"\nESI percentages:")
        for esi, count in esi_counts.items():
            pct = (count / len(df_raw)) * 100
            print(f"  ESI {esi}: {count:,} ({pct:.1f}%)")
    else:
        print("\n❌ 'esi_level' column not found")
        return
    
    # Check RFV1 in original data
    if 'rfv1' in df_raw.columns:
        print(f"\nRFV1 in ORIGINAL data:")
        rfv1_values = df_raw['rfv1'].dropna()
        print(f"  Range: [{rfv1_values.min():.0f}, {rfv1_values.max():.0f}]")
        print(f"  Unique values: {rfv1_values.nunique()}")
        print(f"  Missing values: {df_raw['rfv1'].isnull().sum()} ({df_raw['rfv1'].isnull().sum()/len(df_raw)*100:.1f}%)")
        print(f"  Sample values: {sorted(rfv1_values.unique())[:20]}")
    else:
        print("\n⚠️  'rfv1' column not found")
    
    # Print sample rows
    print(f"\nSample rows (first 10 with key columns):")
    key_cols = ['esi_level', 'age', 'rfv1', 'rfv2', 'rfv3', 'sbp', 'dbp', 'temp_c', 'respiration', 'pain']
    available_cols = [col for col in key_cols if col in df_raw.columns]
    print(df_raw[available_cols].head(10).to_string())
    
    # ============================================================================
    # 2. PREPROCESSED DATA VERIFICATION
    # ============================================================================
    print_section("2. PREPROCESSED DATA VERIFICATION")
    
    cache_v3 = project_root / "services" / "manage-agent" / "outputs" / "preprocessed_data_cache_v3.pkl"
    
    if not cache_v3.exists():
        print(f"❌ Cache v3 not found: {cache_v3}")
        return
    
    print(f"\nLoading preprocessed cache: {cache_v3}")
    with open(cache_v3, 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['train']
    X_val = data['val']
    X_test = data['test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    print(f"\nPreprocessed data shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")
    
    print(f"\nFeature count comparison:")
    print(f"  Original columns (excluding target): {len(df_raw.columns) - 1}")
    print(f"  Preprocessed features: {len(X_train.columns)}")
    print(f"  Features dropped: {len(df_raw.columns) - 1 - len(X_train.columns)}")
    
    # Check ESI distribution in preprocessed data
    print(f"\nESI distribution in PREPROCESSED data:")
    print(f"\nTraining set:")
    train_esi_counts = pd.Series(y_train).value_counts().sort_index()
    for esi, count in train_esi_counts.items():
        pct = (count / len(y_train)) * 100
        print(f"  ESI {esi}: {count:,} ({pct:.1f}%)")
    
    print(f"\nValidation set:")
    val_esi_counts = pd.Series(y_val).value_counts().sort_index()
    for esi, count in val_esi_counts.items():
        pct = (count / len(y_val)) * 100
        print(f"  ESI {esi}: {count:,} ({pct:.1f}%)")
    
    # Check RFV ranges in preprocessed data
    print(f"\nRFV ranges in PREPROCESSED data:")
    rfv_fields = ['rfv1', 'rfv2', 'rfv3', 'rfv1_3d', 'rfv2_3d', 'rfv3_3d']
    for field in rfv_fields:
        if field in X_train.columns:
            values = X_train[field]
            print(f"  {field:10s}: range=[{values.min():.0f}, {values.max():.0f}], mean={values.mean():.0f}, unique={values.nunique()}")
    
    # ============================================================================
    # 3. FEATURE VALUE INSPECTION
    # ============================================================================
    print_section("3. FEATURE VALUE INSPECTION")
    
    print(f"\nAll {len(X_train.columns)} features:")
    print(f"\n{'Feature':<25s} {'Min':<12s} {'Max':<12s} {'Mean':<12s} {'Std':<12s} {'Unique':<10s} {'Variance':<10s}")
    print("-" * 100)
    
    for col in X_train.columns:
        values = X_train[col]
        min_val = values.min()
        max_val = values.max()
        mean_val = values.mean()
        std_val = values.std()
        unique_count = values.nunique()
        variance = values.var()
        
        print(f"{col:<25s} {min_val:<12.4f} {max_val:<12.4f} {mean_val:<12.4f} {std_val:<12.4f} {unique_count:<10d} {variance:<10.4f}")
    
    # Check for features with no variance
    print(f"\nFeatures with no/low variance (potential issues):")
    low_variance_cols = []
    for col in X_train.columns:
        variance = X_train[col].var()
        if variance < 0.01:
            low_variance_cols.append(col)
            print(f"  {col}: variance={variance:.6f}")
    
    if not low_variance_cols:
        print("  ✓ No low-variance features found")
    
    # Correlation with target
    print(f"\nTop 10 features by correlation with target (ESI level):")
    correlations = []
    for col in X_train.columns:
        corr = X_train[col].corr(pd.Series(y_train))
        if not np.isnan(corr):
            correlations.append((col, abs(corr)))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    for col, corr in correlations[:10]:
        print(f"  {col:<25s}: {corr:.4f}")
    
    # Check if features differ across ESI levels
    print(f"\nFeature differences across ESI levels (ANOVA-style check):")
    print(f"\n{'Feature':<25s} {'ESI 0':<12s} {'ESI 1':<12s} {'ESI 2':<12s} {'ESI 3':<12s} {'ESI 4':<12s} {'ESI 5':<12s} {'ESI 6':<12s}")
    print("-" * 100)
    
    for col in X_train.columns[:10]:  # Check first 10 features
        means_by_esi = []
        for esi in sorted(y_train.unique()):
            mask = y_train == esi
            mean_val = X_train[col][mask].mean()
            means_by_esi.append(f"{mean_val:.4f}")
        
        print(f"{col:<25s} {' '.join(means_by_esi)}")
    
    # ============================================================================
    # 4. SANITY CHECKS
    # ============================================================================
    print_section("4. SANITY CHECKS")
    
    # Check if ESI 1 and ESI 2 are present in validation
    print(f"\nESI class presence in validation set:")
    for esi in sorted(y_val.unique()):
        count = (y_val == esi).sum()
        print(f"  ESI {esi}: {count:,} samples ({count/len(y_val)*100:.1f}%)")
    
    # Check train/val/test overlap (data leakage check)
    print(f"\nChecking for data leakage (train/val/test overlap):")
    # Create hash of rows for comparison
    train_hash = pd.util.hash_pandas_object(X_train).sum()
    val_hash = pd.util.hash_pandas_object(X_val).sum()
    test_hash = pd.util.hash_pandas_object(X_test).sum()
    
    if train_hash == val_hash or train_hash == test_hash or val_hash == test_hash:
        print("  ❌ WARNING: Hash collision detected - possible data leakage!")
    else:
        print("  ✓ No hash collisions - train/val/test are distinct")
    
    # Check if we can predict ESI 3 vs ESI 4 (these work in RF)
    print(f"\nBinary classification test: ESI 3 vs ESI 4 (these work in RF):")
    mask_3_4 = (y_train == 3) | (y_train == 4)
    X_train_3_4 = X_train[mask_3_4]
    y_train_3_4 = y_train[mask_3_4]
    
    print(f"  Training samples: {len(X_train_3_4):,}")
    print(f"  ESI 3: {(y_train_3_4 == 3).sum():,}")
    print(f"  ESI 4: {(y_train_3_4 == 4).sum():,}")
    
    # Check feature separability for ESI 3 vs 4
    if len(X_train_3_4) > 0:
        print(f"\n  Feature differences (ESI 3 vs ESI 4):")
        for col in X_train.columns[:5]:  # Check first 5 features
            mean_3 = X_train_3_4[y_train_3_4 == 3][col].mean()
            mean_4 = X_train_3_4[y_train_3_4 == 4][col].mean()
            diff = abs(mean_3 - mean_4)
            print(f"    {col:<25s}: ESI 3={mean_3:.4f}, ESI 4={mean_4:.4f}, diff={diff:.4f}")
    
    # Check distribution similarity
    print(f"\nDistribution similarity check (train vs val):")
    print(f"  Feature means comparison (first 5 features):")
    for col in X_train.columns[:5]:
        train_mean = X_train[col].mean()
        val_mean = X_val[col].mean()
        diff = abs(train_mean - val_mean)
        # Use absolute difference for scaled features (mean≈0) or relative for others
        if abs(train_mean) < 0.1:
            status = "✓" if diff < 0.1 else "⚠️"
            print(f"    {col:<25s}: train={train_mean:.4f}, val={val_mean:.4f}, diff={diff:.4f} {status}")
        else:
            diff_pct = diff / abs(train_mean) * 100
            status = "✓" if diff_pct < 10 else "⚠️"
            print(f"    {col:<25s}: train={train_mean:.4f}, val={val_mean:.4f}, diff={diff_pct:.1f}% {status}")
    
    # ============================================================================
    # 5. RAW DATA SAMPLE INSPECTION
    # ============================================================================
    print_section("5. RAW DATA SAMPLE INSPECTION")
    
    print(f"\nRandom 20 samples showing original features, ESI, and patterns:")
    print("-" * 100)
    
    # Sample 20 random rows
    sample_indices = np.random.choice(len(df_raw), min(20, len(df_raw)), replace=False)
    sample_df = df_raw.iloc[sample_indices].copy()
    
    # Show key features
    key_features = ['esi_level', 'age', 'sex', 'rfv1', 'rfv2', 'rfv3', 'sbp', 'dbp', 'temp_c', 'respiration', 'pain', 'injury']
    available_features = [f for f in key_features if f in sample_df.columns]
    
    print(f"\n{'Index':<8s} {'ESI':<8s} {'Age':<8s} {'Sex':<8s} {'RFV1':<12s} {'SBP':<8s} {'DBP':<8s} {'Temp':<8s} {'Pain':<8s}")
    print("-" * 100)
    
    for idx, row in sample_df.iterrows():
        esi = row.get('esi_level', np.nan)
        age = row.get('age', np.nan)
        sex = row.get('sex', np.nan)
        rfv1 = row.get('rfv1', np.nan)
        sbp = row.get('sbp', np.nan)
        dbp = row.get('dbp', np.nan)
        temp = row.get('temp_c', np.nan)
        pain = row.get('pain', np.nan)
        
        esi_str = f"{esi:.1f}" if not pd.isna(esi) else "N/A"
        age_str = f"{age:.1f}" if not pd.isna(age) else "N/A"
        sex_str = f"{sex:.1f}" if not pd.isna(sex) else "N/A"
        rfv1_str = f"{rfv1:.0f}" if not pd.isna(rfv1) else "N/A"
        sbp_str = f"{sbp:.1f}" if not pd.isna(sbp) else "N/A"
        dbp_str = f"{dbp:.1f}" if not pd.isna(dbp) else "N/A"
        temp_str = f"{temp:.1f}" if not pd.isna(temp) else "N/A"
        pain_str = f"{pain:.1f}" if not pd.isna(pain) else "N/A"
        
        print(f"{idx:<8d} {esi_str:<8s} {age_str:<8s} {sex_str:<8s} {rfv1_str:<12s} {sbp_str:<8s} {dbp_str:<8s} {temp_str:<8s} {pain_str:<8s}")
    
    # Medical sense check
    print(f"\nMedical sense check:")
    print(f"  Checking if ESI 1 (critical) has higher vitals concerns...")
    
    if 'esi_level' in df_raw.columns:
        esi_1_data = df_raw[df_raw['esi_level'] == 1.0]
        esi_3_data = df_raw[df_raw['esi_level'] == 3.0]
        
        if len(esi_1_data) > 0 and len(esi_3_data) > 0:
            print(f"\n  ESI 1 (critical) vs ESI 3 (moderate):")
            for col in ['sbp', 'dbp', 'respiration', 'pain']:
                if col in df_raw.columns:
                    esi_1_mean = esi_1_data[col].mean()
                    esi_3_mean = esi_3_data[col].mean()
                    print(f"    {col:<15s}: ESI 1={esi_1_mean:.2f}, ESI 3={esi_3_mean:.2f}")
    
    # ============================================================================
    # DIAGNOSTIC SUMMARY
    # ============================================================================
    print_section("DIAGNOSTIC SUMMARY & RECOMMENDATIONS", "=")
    
    print("\nKEY FINDINGS:")
    
    # Year quality
    if 'year' in df_raw.columns:
        years = df_raw['year'].unique()
        print(f"\n1. Years in dataset: {sorted(years)}")
    
    # ESI distribution
    print(f"\n2. ESI Distribution:")
    print(f"   Original data: {len(df_raw)} samples")
    print(f"   Training data: {len(y_train)} samples (after SMOTE)")
    print(f"   Validation data: {len(y_val)} samples")
    
    # Feature quality
    print(f"\n3. Feature Quality:")
    print(f"   Total features: {len(X_train.columns)}")
    print(f"   Low variance features: {len(low_variance_cols)}")
    if correlations:
        top_corr = correlations[0]
        print(f"   Highest correlation with target: {top_corr[0]} ({top_corr[1]:.4f})")
    
    # Class imbalance
    print(f"\n4. Class Imbalance:")
    train_esi_counts = pd.Series(y_train).value_counts().sort_index()
    minority_classes = [esi for esi, count in train_esi_counts.items() if count < len(y_train) * 0.05]
    if minority_classes:
        print(f"   ⚠️  Minority classes (<5%): {minority_classes}")
    else:
        print(f"   ✓ Classes are balanced (after SMOTE)")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    print(f"\n1. Check if ESI labels are correct in original data")
    print(f"2. Verify feature engineering (RFV codes, vitals, etc.)")
    print(f"3. Consider stratified sampling for minority classes")
    print(f"4. Check if problem is inherently difficult (7 classes, overlapping features)")
    print(f"5. Consider feature selection or dimensionality reduction")
    
    print_section("END OF DIAGNOSTIC", "=")


if __name__ == "__main__":
    main()

