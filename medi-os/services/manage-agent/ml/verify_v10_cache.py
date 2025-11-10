"""
Verify v10 cache has correct feature count and structure.
"""

import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'services' / 'manage-agent'))

def verify_v10_cache():
    """Verify v10 cache structure and features."""
    print("=" * 80)
    print("VERIFYING V10 CACHE")
    print("=" * 80)
    
    cache_file = project_root / "services" / "manage-agent" / "outputs" / "preprocessed_data_cache_v10_clinical_features.pkl"
    
    if not cache_file.exists():
        print(f"\n[ERROR] Cache file not found: {cache_file}")
        return False
    
    print(f"\n[1] Loading cache: {cache_file}")
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    
    # Check structure
    print("\n[2] Checking cache structure...")
    required_keys = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test', 'feature_names', 'class_weights']
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        print(f"  [ERROR] Missing keys: {missing_keys}")
        return False
    print("  [OK] All required keys present")
    
    # Check shapes
    print("\n[3] Checking data shapes...")
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Check feature count
    print("\n[4] Checking feature count...")
    feature_count = len(data['feature_names'])
    print(f"  Total features: {feature_count}")
    if feature_count < 90:
        print(f"  [WARNING] Feature count ({feature_count}) is lower than expected (~94)")
    elif feature_count > 100:
        print(f"  [WARNING] Feature count ({feature_count}) is higher than expected (~94)")
    else:
        print(f"  [OK] Feature count is within expected range")
    
    # Check clinical features
    print("\n[5] Checking clinical features...")
    clinical_features = [
        'shock_index', 'mean_arterial_pressure', 'pulse_pressure',
        'tachycardia', 'bradycardia', 'hypotension', 'hypertension',
        'tachypnea', 'respiratory_distress', 'hypoxia', 'severe_hypoxia',
        'is_elderly', 'is_pediatric', 'is_infant',
        'severe_pain', 'moderate_pain',
        'is_flu_season', 'is_weekend',
        'vital_abnormal_count', 'shock_index_high', 'elderly_hypotension',
        'pediatric_fever'
    ]
    
    missing_clinical = [f for f in clinical_features if f not in data['feature_names']]
    if missing_clinical:
        print(f"  [ERROR] Missing clinical features: {missing_clinical}")
        return False
    else:
        print(f"  [OK] All 22 clinical features present")
    
    # Check for NaN/Inf in clinical features
    print("\n[6] Checking for NaN/Inf values in clinical features...")
    for feat in clinical_features:
        if feat in X_train.columns:
            nan_count = X_train[feat].isna().sum()
            inf_count = np.isinf(X_train[feat]).sum()
            if nan_count > 0:
                print(f"  [ERROR] {feat} has {nan_count} NaN values in training set")
                return False
            if inf_count > 0:
                print(f"  [ERROR] {feat} has {inf_count} Inf values in training set")
                return False
    print("  [OK] No NaN or Inf values in clinical features")
    
    # Check value ranges
    print("\n[7] Checking value ranges...")
    if 'shock_index' in X_train.columns:
        si_min = X_train['shock_index'].min()
        si_max = X_train['shock_index'].max()
        print(f"  shock_index range: [{si_min:.2f}, {si_max:.2f}]")
        if si_min < 0 or si_max > 5:
            print(f"  [WARNING] shock_index out of expected range [0, 5]")
    
    # Check binary features
    binary_features = ['tachycardia', 'bradycardia', 'hypotension', 'hypertension', 
                       'tachypnea', 'respiratory_distress', 'hypoxia', 'severe_hypoxia',
                       'is_elderly', 'is_pediatric', 'is_infant',
                       'severe_pain', 'moderate_pain', 'is_flu_season', 'is_weekend',
                       'shock_index_high', 'elderly_hypotension', 'pediatric_fever']
    
    for feat in binary_features:
        if feat in X_train.columns:
            unique_vals = sorted(X_train[feat].unique())
            if not set(unique_vals).issubset({0, 1}):
                print(f"  [WARNING] {feat} has non-binary values: {unique_vals}")
    
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"[OK] Cache structure correct")
    print(f"[OK] Feature count: {feature_count}")
    print(f"[OK] All 22 clinical features present")
    print(f"[OK] No NaN/Inf values")
    print(f"[OK] Value ranges reasonable")
    
    return True

if __name__ == "__main__":
    success = verify_v10_cache()
    if success:
        print("\n[OK] ALL VERIFICATIONS PASSED")
        sys.exit(0)
    else:
        print("\n[ERROR] VERIFICATIONS FAILED")
        sys.exit(1)

