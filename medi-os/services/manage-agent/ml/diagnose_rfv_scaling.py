"""
Diagnose RFV scaling issue.

CRITICAL: RFV codes (categorical numeric: 1000-15000) should NOT be standardized.
StandardScaler destroys their categorical meaning.
"""

import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'services' / 'manage-agent'))


def main():
    print("=" * 60)
    print("RFV SCALING DIAGNOSTIC")
    print("=" * 60)
    print("\nChecking if RFV codes were incorrectly standardized...")
    
    # Load preprocessed data (try v3 first, then v2)
    cache_v3 = project_root / "services" / "manage-agent" / "outputs" / "preprocessed_data_cache_v3.pkl"
    cache_v2 = project_root / "services" / "manage-agent" / "outputs" / "preprocessed_data_cache_v2.pkl"
    
    if cache_v3.exists():
        cache_file = cache_v3
        print(f"\nLoading v3 cache: {cache_file}")
    elif cache_v2.exists():
        cache_file = cache_v2
        print(f"\nLoading v2 cache: {cache_file}")
    else:
        print(f"\n❌ Cache files not found")
        return
    
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['train']
    
    # RFV fields to check
    rfv_fields = ['rfv1', 'rfv2', 'rfv3', 'rfv1_3d', 'rfv2_3d', 'rfv3_3d']
    
    print(f"\n" + "=" * 60)
    print("RFV VALUE RANGES (AFTER PREPROCESSING)")
    print("=" * 60)
    
    print("\nExpected: RFV codes should be in range 1000-15000 (original codes)")
    print("Actual: Check ranges below...")
    
    rfv_scaled = {}
    
    for field in rfv_fields:
        if field in X_train.columns:
            values = X_train[field]
            min_val = values.min()
            max_val = values.max()
            mean_val = values.mean()
            std_val = values.std()
            unique_count = values.nunique()
            
            # Check if scaled (mean≈0, std≈1, range in [-3, 3])
            is_scaled = abs(mean_val) < 0.5 and abs(std_val - 1.0) < 0.5 and abs(min_val) < 3 and abs(max_val) < 3
            
            status = "❌ SCALED (WRONG!)" if is_scaled else "✅ Original range"
            rfv_scaled[field] = is_scaled
            
            print(f"\n  {field}:")
            print(f"    Range: [{min_val:.2f}, {max_val:.2f}]")
            print(f"    Mean:  {mean_val:.2f}")
            print(f"    Std:   {std_val:.2f}")
            print(f"    Unique values: {unique_count}")
            print(f"    Status: {status}")
    
    # Check raw data for comparison
    print(f"\n" + "=" * 60)
    print("CHECKING RAW DATA (FOR COMPARISON)")
    print("=" * 60)
    
    raw_data_path = project_root / "data" / "NHAMCS_2011_2022_combined.csv"
    
    if raw_data_path.exists():
        print(f"\nLoading raw data: {raw_data_path}")
        df_raw = pd.read_csv(raw_data_path)
        
        print("\nRaw RFV ranges (BEFORE preprocessing):")
        for field in rfv_fields:
            if field in df_raw.columns:
                values = df_raw[field].dropna()
                if len(values) > 0:
                    min_val = values.min()
                    max_val = values.max()
                    unique_count = values.nunique()
                    print(f"  {field}: range=[{min_val:.0f}, {max_val:.0f}], unique={unique_count}")
    
    # Final diagnosis
    print(f"\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    
    scaled_count = sum(rfv_scaled.values())
    
    if scaled_count > 0:
        print(f"\n❌ CRITICAL ISSUE FOUND!")
        print(f"   {scaled_count} RFV field(s) were incorrectly standardized")
        print(f"   This destroys their categorical meaning!")
        print(f"\n   Impact:")
        print(f"   - RFV codes lost their interpretability")
        print(f"   - Model can't learn meaningful patterns")
        print(f"   - Feature importance drops (current: 0.047)")
        print(f"   - Accuracy stuck at ~50%")
        print(f"\n   Solution:")
        print(f"   - Re-preprocess WITHOUT scaling RFV codes")
        print(f"   - Only scale numeric features (age, vitals)")
        print(f"   - Keep RFV in original range (1000-15000)")
    else:
        print(f"\n✅ RFV codes are in original range (not scaled)")
        print(f"   This is correct!")
    
    print(f"\n" + "=" * 60)


if __name__ == "__main__":
    main()

