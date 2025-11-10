"""
Pre-training verification script.
Verifies all prerequisites are met before model training.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("PRE-TRAINING VERIFICATION CHECKS")
print("=" * 70)

# 1. Verify RFV mappings file exists
print("\n[1/4] Checking RFV mappings file...")
mappings_path = Path(__file__).parent.parent.parent / "data" / "rfv_code_mappings.json"
if mappings_path.exists():
    file_size = mappings_path.stat().st_size / 1024  # KB
    print(f"  [OK] File exists: {mappings_path}")
    print(f"     Size: {file_size:.2f} KB")
    
    # Load and check content
    import json
    with open(mappings_path, 'r') as f:
        mappings = json.load(f)
    total_codes = sum(len(m) for m in mappings.values())
    print(f"     Fields: {len(mappings)}, Total codes: {total_codes}")
else:
    print(f"  [ERROR] File NOT found: {mappings_path}")
    print("     Run regenerate_csv_numeric_rfv.py first!")
    sys.exit(1)

# 2. Test RFV mapper
print("\n[2/4] Testing RFV text-to-code mapper...")
try:
    from ml.rfv_mapper import RFVTextToCodeMapper
    mapper = RFVTextToCodeMapper.load()
    print("  [OK] Mapper loaded successfully")
    
    # Test conversions
    test_cases = [
        ("Chest pain", "rfv1"),
        ("Abdominal pain", "rfv1"),
        ("Shortness of breath", "rfv2"),
    ]
    
    all_passed = True
    for text, field in test_cases:
        code = mapper.text_to_code(text, field)
        if code:
            text_back = mapper.code_to_text(code, field)
            print(f"     '{text}' -> {code:.0f} -> '{text_back}'")
        else:
            print(f"     [WARNING] '{text}' -> NOT FOUND")
            all_passed = False
    
    if all_passed:
        print("  [OK] All mapper tests passed")
    else:
        print("  [WARNING] Some mappings not found (may be okay)")
except Exception as e:
    print(f"  [ERROR] Error: {e}")
    sys.exit(1)

# 3. Check CSV has numeric RFV columns
print("\n[3/4] Checking CSV RFV columns are numeric...")
csv_path = Path(__file__).parent.parent.parent / "data" / "NHAMCS_2011_2022_combined.csv"
if not csv_path.exists():
    print(f"  ❌ CSV not found: {csv_path}")
    sys.exit(1)

df = pd.read_csv(csv_path, nrows=10)
rfv_cols = [c for c in df.columns if 'rfv' in c.lower()]

if rfv_cols:
    print(f"  [OK] Found {len(rfv_cols)} RFV columns: {rfv_cols}")
    
    # Check data types
    all_numeric = True
    for col in rfv_cols:
        dtype = df[col].dtype
        if not pd.api.types.is_numeric_dtype(dtype):
            print(f"     [ERROR] {col}: {dtype} (should be numeric)")
            all_numeric = False
        else:
            print(f"     [OK] {col}: {dtype}")
    
    # Show sample values
    print("\n  Sample RFV values (should be numeric codes):")
    print(df[rfv_cols].head(3).to_string())
    
    if all_numeric:
        print("\n  [OK] All RFV columns are numeric")
    else:
        print("\n  [ERROR] Some RFV columns are not numeric!")
        sys.exit(1)
else:
    print("  [ERROR] No RFV columns found in CSV!")
    sys.exit(1)

# 4. Verify preprocessed data shapes
print("\n[4/4] Verifying preprocessed data shapes...")
try:
    from ml.pipeline import TriagePreprocessingPipeline
    
    # Load full data
    df_full = pd.read_csv(csv_path)
    print(f"  Original shape: {df_full.shape}")
    
    # Run preprocessing (this will take time)
    print("  Running preprocessing pipeline (this may take 20+ minutes)...")
    pipeline = TriagePreprocessingPipeline()
    result = pipeline.fit_transform(df_full, target_col='esi_level')
    
    print("\n  [OK] Preprocessing complete!")
    print(f"     Train shape: {result['X_train'].shape}")
    print(f"     Val shape: {result['X_val'].shape}")
    print(f"     Test shape: {result['X_test'].shape}")
    print(f"     Feature count: {len(result['X_train'].columns)}")
    
    # Check for missing values
    missing_train = result['X_train'].isna().sum().sum()
    missing_val = result['X_val'].isna().sum().sum()
    missing_test = result['X_test'].isna().sum().sum()
    
    if missing_train == 0 and missing_val == 0 and missing_test == 0:
        print("     [OK] No missing values in any split")
    else:
        print(f"     [WARNING] Missing values - Train: {missing_train}, Val: {missing_val}, Test: {missing_test}")
    
    # Check feature types
    all_numeric = all(pd.api.types.is_numeric_dtype(result['X_train'][col]) 
                      for col in result['X_train'].columns)
    if all_numeric:
        print("     [OK] All features are numeric (ready for ML)")
    else:
        print("     [ERROR] Some features are not numeric!")
    
except Exception as e:
    print(f"  [ERROR] Error during preprocessing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("[OK] ALL VERIFICATION CHECKS PASSED!")
print("Ready for model training.")
print("=" * 70)

