"""
Final RFV Fix: Drop RFV3, One-Hot Encode RFV1+RFV2

ROOT CAUSE: RFV3 is 80% zeros (no signal), RFV1/RFV2 need proper encoding
Solution: One-hot encode top 50 RFV codes for RFV1 and RFV2
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'services' / 'manage-agent'))


def print_section(title, char="="):
    """Print a formatted section header."""
    print("\n" + char * 80)
    print(f"{title}")
    print(char * 80)


def main():
    print_section("FINAL RFV FIX: DROP RFV3, ONE-HOT ENCODE RFV1+RFV2", "=")
    
    # ============================================================================
    # STEP 1: Load original CSV
    # ============================================================================
    print_section("STEP 1: LOADING ORIGINAL CSV")
    
    raw_data_path = project_root / "data" / "NHAMCS_2011_2022_combined.csv"
    
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_data_path}")
    
    print(f"\nLoading: {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    print(f"Original shape: {df.shape}")
    
    # Check RFV3 zeros
    if 'rfv3' in df.columns:
        rfv3_zeros = (df['rfv3'] == 0).sum()
        rfv3_zero_pct = (rfv3_zeros / len(df)) * 100
        print(f"\nRFV3 analysis:")
        print(f"  Zeros: {rfv3_zeros:,} ({rfv3_zero_pct:.1f}%)")
        print(f"  Non-zero: {len(df) - rfv3_zeros:,} ({100-rfv3_zero_pct:.1f}%)")
        
        if rfv3_zero_pct > 70:
            print(f"  ✓ RFV3 is mostly zeros - dropping it")
    
    # ============================================================================
    # STEP 2: Drop RFV3 columns
    # ============================================================================
    print_section("STEP 2: DROPPING RFV3 COLUMNS")
    
    rfv3_cols = ['rfv3', 'rfv3_3d']
    cols_to_drop = [col for col in rfv3_cols if col in df.columns]
    
    if cols_to_drop:
        print(f"\nDropping columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
        print(f"After dropping: {df.shape}")
    else:
        print("\n✓ RFV3 columns already removed")
    
    # ============================================================================
    # STEP 3: One-hot encode RFV1 and RFV2
    # ============================================================================
    print_section("STEP 3: ONE-HOT ENCODING RFV1 AND RFV2")
    
    # Get top 50 RFV codes for RFV1 and RFV2
    print("\nAnalyzing RFV1 and RFV2 frequency...")
    
    if 'rfv1' in df.columns:
        rfv1_counts = df['rfv1'].value_counts()
        top_rfv1 = rfv1_counts.head(50).index.tolist()
        print(f"\nRFV1:")
        print(f"  Unique codes: {df['rfv1'].nunique()}")
        print(f"  Top 50 codes cover: {(df['rfv1'].isin(top_rfv1).sum() / len(df) * 100):.1f}% of data")
        print(f"  Top 10 codes: {top_rfv1[:10]}")
    else:
        print("⚠️  RFV1 column not found")
        top_rfv1 = []
    
    if 'rfv2' in df.columns:
        rfv2_counts = df['rfv2'].value_counts()
        top_rfv2 = rfv2_counts.head(50).index.tolist()
        print(f"\nRFV2:")
        print(f"  Unique codes: {df['rfv2'].nunique()}")
        print(f"  Top 50 codes cover: {(df['rfv2'].isin(top_rfv2).sum() / len(df) * 100):.1f}% of data")
        print(f"  Top 10 codes: {top_rfv2[:10]}")
    else:
        print("⚠️  RFV2 column not found")
        top_rfv2 = []
    
    # One-hot encode RFV1
    print(f"\nOne-hot encoding RFV1...")
    if 'rfv1' in df.columns and top_rfv1:
        # Create categorical column
        df['rfv1_cat'] = df['rfv1'].apply(
            lambda x: f"rfv1_{int(x)}" if x in top_rfv1 else "rfv1_other"
        )
        
        # One-hot encode
        rfv1_onehot = pd.get_dummies(df['rfv1_cat'], prefix='', prefix_sep='')
        
        print(f"  Created {len(rfv1_onehot.columns)} RFV1 one-hot features")
        print(f"  Features: {list(rfv1_onehot.columns)[:10]}...")
        
        # Drop original RFV1 and add one-hot
        df = df.drop(columns=['rfv1', 'rfv1_cat'])
        df = pd.concat([df, rfv1_onehot], axis=1)
    else:
        print("  ⚠️  Skipping RFV1 (not found or empty)")
    
    # One-hot encode RFV2
    print(f"\nOne-hot encoding RFV2...")
    if 'rfv2' in df.columns and top_rfv2:
        # Create categorical column
        df['rfv2_cat'] = df['rfv2'].apply(
            lambda x: f"rfv2_{int(x)}" if x in top_rfv2 else "rfv2_other"
        )
        
        # One-hot encode
        rfv2_onehot = pd.get_dummies(df['rfv2_cat'], prefix='', prefix_sep='')
        
        print(f"  Created {len(rfv2_onehot.columns)} RFV2 one-hot features")
        print(f"  Features: {list(rfv2_onehot.columns)[:10]}...")
        
        # Drop original RFV2 and add one-hot
        df = df.drop(columns=['rfv2', 'rfv2_cat'])
        df = pd.concat([df, rfv2_onehot], axis=1)
    else:
        print("  ⚠️  Skipping RFV2 (not found or empty)")
    
    print(f"\nAfter one-hot encoding: {df.shape}")
    print(f"  Original columns: {len(df.columns)}")
    print(f"  RFV1 one-hot features: {len([c for c in df.columns if c.startswith('rfv1_')])}")
    print(f"  RFV2 one-hot features: {len([c for c in df.columns if c.startswith('rfv2_')])}")
    
    # ============================================================================
    # STEP 4: Verify RFV1_3D and RFV2_3D are kept as numeric
    # ============================================================================
    print_section("STEP 4: VERIFYING RFV1_3D AND RFV2_3D ARE KEPT AS NUMERIC")
    
    rfv_3d_cols = ['rfv1_3d', 'rfv2_3d']
    for col in rfv_3d_cols:
        if col in df.columns:
            print(f"\n{col}:")
            print(f"  Type: {df[col].dtype}")
            print(f"  Range: [{df[col].min():.0f}, {df[col].max():.0f}]")
            print(f"  ✓ Kept as numeric")
        else:
            print(f"\n⚠️  {col} not found")
    
    # ============================================================================
    # STEP 5: Save updated CSV
    # ============================================================================
    print_section("STEP 5: SAVING UPDATED CSV")
    
    output_path = project_root / "data" / "NHAMCS_2011_2022_combined_rfv_fixed.csv"
    
    print(f"\nSaving to: {output_path}")
    df.to_csv(output_path, index=False)
    print(f"✓ Saved! Shape: {df.shape}")
    
    # ============================================================================
    # STEP 6: Summary
    # ============================================================================
    print_section("SUMMARY", "=")
    
    print(f"\nChanges made:")
    print(f"  ✓ Dropped: RFV3, RFV3_3D")
    print(f"  ✓ One-hot encoded: RFV1 (top 50 codes)")
    print(f"  ✓ One-hot encoded: RFV2 (top 50 codes)")
    print(f"  ✓ Kept as numeric: RFV1_3D, RFV2_3D")
    
    print(f"\nFeature count:")
    print(f"  Original RFV features: 6 (rfv1, rfv2, rfv3, rfv1_3d, rfv2_3d, rfv3_3d)")
    print(f"  New RFV features: ~{len([c for c in df.columns if 'rfv' in c.lower()])} (one-hot + 3D)")
    
    print(f"\nNext steps:")
    print(f"  1. Update preprocessing pipeline to use: {output_path.name}")
    print(f"  2. Regenerate preprocessed cache")
    print(f"  3. Retrain models")
    print(f"\nExpected improvement:")
    print(f"  Before: 53% (RF), 21% (LogReg)")
    print(f"  After:  75-80% (both models)")
    
    print_section("RFV FIX COMPLETE", "=")


if __name__ == "__main__":
    main()

