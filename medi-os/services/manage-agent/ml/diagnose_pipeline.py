"""
Diagnostic script to identify issues causing poor model performance.

Expected: 92-93% accuracy
Actual: 49% accuracy (random chance)

This script checks for common data pipeline issues.
"""

import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'services' / 'manage-agent'))


def check_data_sanity(X_train, X_val, X_test, y_train, y_val, y_test):
    """Check 1: Data sanity checks."""
    print("\n" + "=" * 60)
    print("CHECK 1: DATA SANITY")
    print("=" * 60)
    
    # Shapes
    # Allow flexible feature count (v2 has 23 features, v1 had 34)
    expected_train_samples = 382039
    expected_val_samples = 26824
    expected_test_samples = 26825
    expected_features = len(X_train.columns)  # Use actual feature count
    
    print(f"\nShapes:")
    print(f"  Train: {X_train.shape} (expected: ({expected_train_samples}, {expected_features}))")
    print(f"  Val:   {X_val.shape} (expected: ({expected_val_samples}, {expected_features}))")
    print(f"  Test:  {X_test.shape} (expected: ({expected_test_samples}, {expected_features}))")
    
    shape_ok = (
        X_train.shape[0] == expected_train_samples and
        X_val.shape[0] == expected_val_samples and
        X_test.shape[0] == expected_test_samples and
        X_train.shape[1] == X_val.shape[1] == X_test.shape[1] == expected_features
    )
    print(f"  ✅ Shapes correct" if shape_ok else f"  ❌ Shapes MISMATCH!")
    
    # NaN values
    train_nan = X_train.isnull().sum().sum()
    val_nan = X_val.isnull().sum().sum()
    test_nan = X_test.isnull().sum().sum()
    
    print(f"\nNaN values:")
    print(f"  Train: {train_nan} (expected: 0)")
    print(f"  Val:   {val_nan} (expected: 0)")
    print(f"  Test:  {test_nan} (expected: 0)")
    
    nan_ok = train_nan == 0 and val_nan == 0 and test_nan == 0
    print(f"  ✅ No NaN values" if nan_ok else f"  ❌ NaN values found!")
    
    # Inf values
    train_inf = np.isinf(X_train.select_dtypes(include=[np.number]).values).sum()
    val_inf = np.isinf(X_val.select_dtypes(include=[np.number]).values).sum()
    test_inf = np.isinf(X_test.select_dtypes(include=[np.number]).values).sum()
    
    print(f"\nInf values:")
    print(f"  Train: {train_inf} (expected: 0)")
    print(f"  Val:   {val_inf} (expected: 0)")
    print(f"  Test:  {test_inf} (expected: 0)")
    
    inf_ok = train_inf == 0 and val_inf == 0 and test_inf == 0
    print(f"  ✅ No Inf values" if inf_ok else f"  ❌ Inf values found!")
    
    # Constant columns
    constant_cols = []
    for col in X_train.columns:
        if X_train[col].nunique() <= 1:
            constant_cols.append(col)
    
    print(f"\nConstant columns (all same value): {len(constant_cols)}")
    if constant_cols:
        print(f"  ❌ Constant columns found: {constant_cols[:5]}")
    else:
        print(f"  ✅ No constant columns")
    
    # Feature value ranges
    print(f"\nFeature value ranges (sample of 5 features):")
    sample_cols = X_train.columns[:5]
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    for col in sample_cols:
        if col in numeric_cols:
            col_data = X_train[col]
            mean_val = col_data.mean()
            std_val = col_data.std()
            min_val = col_data.min()
            max_val = col_data.max()
            print(f"  {col:20s}: mean={mean_val:8.2f}, std={std_val:8.2f}, range=[{min_val:.2f}, {max_val:.2f}]")
        else:
            print(f"  {col:20s}: (non-numeric)")
    
    return shape_ok and nan_ok and inf_ok and len(constant_cols) == 0


def check_target_encoding(y_train, y_val, y_test):
    """Check 2: Target encoding verification."""
    print("\n" + "=" * 60)
    print("CHECK 2: TARGET ENCODING")
    print("=" * 60)
    
    # Get unique values
    train_unique = sorted(y_train.unique())
    val_unique = sorted(y_val.unique())
    test_unique = sorted(y_test.unique())
    
    print(f"\nUnique ESI levels:")
    print(f"  Train: {train_unique}")
    print(f"  Val:   {val_unique}")
    print(f"  Test:  {test_unique}")
    
    # Check if correctly encoded (should be [0,1,2,3,4,5,6])
    expected_encoded = [0, 1, 2, 3, 4, 5, 6]
    train_encoded_correct = train_unique == expected_encoded
    val_encoded_correct = val_unique == expected_encoded
    test_encoded_correct = test_unique == expected_encoded
    
    print(f"\nExpected encoded: [0, 1, 2, 3, 4, 5, 6]")
    print(f"  Train: {'✅ Correctly encoded' if train_encoded_correct else '❌ MISMATCH!'}")
    print(f"  Val:   {'✅ Correctly encoded' if val_encoded_correct else '❌ MISMATCH!'}")
    print(f"  Test:  {'✅ Correctly encoded' if test_encoded_correct else '❌ MISMATCH!'}")
    
    # Check for original ESI 7 (should NOT be present)
    train_has_7 = 7 in train_unique or 7.0 in train_unique
    val_has_7 = 7 in val_unique or 7.0 in val_unique
    test_has_7 = 7 in test_unique or 7.0 in test_unique
    
    print(f"\nOriginal ESI 7 check:")
    print(f"  Train: {'❌ Has ESI 7 (should be 6)' if train_has_7 else '✅ No ESI 7'}")
    print(f"  Val:   {'❌ Has ESI 7 (should be 6)' if val_has_7 else '✅ No ESI 7'}")
    print(f"  Test:  {'❌ Has ESI 7 (should be 6)' if test_has_7 else '✅ No ESI 7'}")
    
    # Check if ESI 6 exists (it should after mapping)
    train_has_6 = 6 in train_unique or 6.0 in train_unique
    val_has_6 = 6 in val_unique or 6.0 in val_unique
    test_has_6 = 6 in test_unique or 6.0 in test_unique
    
    print(f"\nESI 6 check (should exist after mapping):")
    print(f"  Train: {'✅ Has ESI 6' if train_has_6 else '❌ Missing ESI 6!'}")
    print(f"  Val:   {'✅ Has ESI 6' if val_has_6 else '❌ Missing ESI 6!'}")
    print(f"  Test:  {'✅ Has ESI 6' if test_has_6 else '❌ Missing ESI 6!'}")
    
    return train_encoded_correct and val_encoded_correct and test_encoded_correct and not train_has_7 and not val_has_7 and not test_has_7 and train_has_6


def check_class_distribution(y_train, y_val, y_test):
    """Check 3: Class distribution."""
    print("\n" + "=" * 60)
    print("CHECK 3: CLASS DISTRIBUTION")
    print("=" * 60)
    
    train_counts = Counter(y_train)
    val_counts = Counter(y_val)
    test_counts = Counter(y_test)
    
    print(f"\nTrain class distribution (should be balanced ~54,577 each):")
    for cls in sorted(train_counts.keys()):
        count = train_counts[cls]
        pct = (count / len(y_train)) * 100
        expected = 54577
        diff_pct = abs(count - expected) / expected * 100
        status = "✅" if diff_pct < 5 else "⚠️"
        print(f"  Class {cls}: {count:6,} ({pct:5.1f}%) {status}")
    
    print(f"\nVal class distribution (should be imbalanced, original):")
    for cls in sorted(val_counts.keys()):
        count = val_counts[cls]
        pct = (count / len(y_val)) * 100
        print(f"  Class {cls}: {count:6,} ({pct:5.1f}%)")
    
    print(f"\nTest class distribution (should be imbalanced, original):")
    for cls in sorted(test_counts.keys()):
        count = test_counts[cls]
        pct = (count / len(y_test)) * 100
        print(f"  Class {cls}: {count:6,} ({pct:5.1f}%)")
    
    # Check if train is balanced
    train_counts_list = list(train_counts.values())
    train_balanced = max(train_counts_list) / min(train_counts_list) < 1.1  # Within 10%
    
    print(f"\nTrain balance check:")
    print(f"  {'✅ Train is balanced (SMOTE worked)' if train_balanced else '❌ Train is NOT balanced (SMOTE issue)'}")
    
    # Check if val is imbalanced (should be)
    val_counts_list = list(val_counts.values())
    val_imbalanced = max(val_counts_list) / min(val_counts_list) > 2  # At least 2x difference
    
    print(f"  {'✅ Val is imbalanced (correct)' if val_imbalanced else '⚠️ Val looks balanced (possible SMOTE leakage)'}")
    
    return train_balanced and val_imbalanced


def check_smote_leakage(X_train, X_val, X_test, y_train, y_val, y_test):
    """Check 4: SMOTE leakage (critical bug if found)."""
    print("\n" + "=" * 60)
    print("CHECK 4: SMOTE LEAKAGE CHECK")
    print("=" * 60)
    
    # Check val size
    expected_val_size = 26824
    val_size_ok = len(X_val) == expected_val_size
    
    print(f"\nValidation set size:")
    print(f"  Actual:   {len(X_val):,}")
    print(f"  Expected: {expected_val_size:,}")
    print(f"  {'✅ Correct size' if val_size_ok else '❌ WRONG SIZE (SMOTE applied to val?)'}")
    
    # Check for duplicates between train and val
    # Convert to tuples for comparison (sample if too large)
    print(f"\nChecking for train-val duplicates...")
    sample_size = min(10000, len(X_val))
    val_sample = X_val.iloc[:sample_size].values
    train_sample = X_train.sample(min(50000, len(X_train))).values
    
    # Check if any val rows match train rows
    duplicates_found = 0
    for i in range(len(val_sample)):
        if i % 1000 == 0:
            print(f"  Checking row {i}/{sample_size}...", end='\r')
        if np.any(np.all(train_sample == val_sample[i], axis=1)):
            duplicates_found += 1
    
    print(f"\n  Duplicates found in sample: {duplicates_found}")
    print(f"  {'✅ No duplicates (no SMOTE leakage)' if duplicates_found == 0 else '❌ DUPLICATES FOUND (SMOTE LEAKAGE!)'}")
    
    # Check if val has synthetic pattern (SMOTE creates synthetic samples)
    # If SMOTE was applied incorrectly, val might have very similar samples
    val_near_duplicates = 0
    if len(X_val) > 1000:
        val_sample = X_val.sample(1000).values
        for i in range(len(val_sample)):
            for j in range(i+1, len(val_sample)):
                if np.allclose(val_sample[i], val_sample[j], rtol=1e-5):
                    val_near_duplicates += 1
                    break
    
    print(f"\n  Near-duplicates in val (within 1e-5): {val_near_duplicates}")
    print(f"  {'✅ No suspicious patterns' if val_near_duplicates < 10 else '⚠️ Many near-duplicates (possible SMOTE leakage)'}")
    
    return val_size_ok and duplicates_found == 0


def check_preprocessing_scale(X_train, X_val, X_test):
    """Check 5: Preprocessing scale consistency."""
    print("\n" + "=" * 60)
    print("CHECK 5: PREPROCESSING SCALE")
    print("=" * 60)
    print("  Note: StandardScaler fits on train (mean≈0, std≈1)")
    print("        Val/test transformed with same scaler, may have different means/stds")
    print("        This is CORRECT behavior - different distributions = different transformed stats")
    
    # Get numeric columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    
    print(f"\nScale comparison (mean/std) for sample features:")
    sample_cols = numeric_cols[:5]
    
    scale_issues = []
    for col in sample_cols:
        train_mean = X_train[col].mean()
        train_std = X_train[col].std()
        val_mean = X_val[col].mean()
        val_std = X_val[col].std()
        test_mean = X_test[col].mean()
        test_std = X_test[col].std()
        
        # Check if train is properly scaled (mean≈0, std≈1)
        train_scaled_ok = abs(train_mean) < 0.1 and abs(train_std - 1.0) < 0.1
        
        # Check if val/test are transformed (std should be close to 1, mean can vary)
        val_scaled_ok = abs(val_std - 1.0) < 0.2  # Allow ±0.2 std difference
        test_scaled_ok = abs(test_std - 1.0) < 0.2
        
        # Overall check: train must be scaled correctly, val/test stds reasonable
        col_ok = train_scaled_ok and val_scaled_ok and test_scaled_ok
        
        status = "✅" if col_ok else "⚠️"
        if not col_ok:
            if not train_scaled_ok:
                scale_issues.append(f"{col} (train not scaled)")
            elif not val_scaled_ok or not test_scaled_ok:
                scale_issues.append(f"{col} (val/test std unusual)")
        
        print(f"\n  {col}:")
        print(f"    Train: mean={train_mean:8.4f}, std={train_std:8.4f} {'✅' if train_scaled_ok else '❌'}")
        print(f"    Val:   mean={val_mean:8.4f}, std={val_std:8.4f} {'✅' if val_scaled_ok else '⚠️'}")
        print(f"    Test:  mean={test_mean:8.4f}, std={test_std:8.4f} {'✅' if test_scaled_ok else '⚠️'}")
    
    if scale_issues:
        print(f"\n  ⚠️  Issues found: {scale_issues[:3]}")
        print(f"     Note: Val/test means/stds differ from train because they have different distributions")
        print(f"     This is EXPECTED - StandardScaler transforms all sets with train's parameters")
    else:
        print(f"\n  ✅ StandardScaler applied correctly")
        print(f"     Train: mean≈0, std≈1 (correct)")
        print(f"     Val/test: transformed with same scaler (correct)")
    
    # Check if train is properly scaled (this is the critical check)
    train_properly_scaled = all(
        abs(X_train[col].mean()) < 0.1 and abs(X_train[col].std() - 1.0) < 0.1
        for col in numeric_cols[:10]  # Check first 10 numeric columns
    )
    
    return train_properly_scaled


def check_baseline_comparison(y_val, model_accuracies):
    """Check 6: Baseline comparison."""
    print("\n" + "=" * 60)
    print("CHECK 6: BASELINE COMPARISON")
    print("=" * 60)
    
    # Calculate majority class baseline
    val_counts = Counter(y_val)
    majority_class = max(val_counts, key=val_counts.get)
    majority_count = val_counts[majority_class]
    baseline_accuracy = majority_count / len(y_val)
    
    print(f"\nMajority class baseline:")
    print(f"  Majority class: {majority_class}")
    print(f"  Count: {majority_count:,} / {len(y_val):,}")
    print(f"  Baseline accuracy: {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
    
    print(f"\nModel accuracies vs baseline:")
    for model_name, acc in model_accuracies.items():
        diff = acc - baseline_accuracy
        status = "✅" if acc > baseline_accuracy else "❌"
        print(f"  {model_name:20s}: {acc:.4f} (diff: {diff:+.4f}) {status}")
        if acc < baseline_accuracy:
            print(f"    ⚠️  MODEL WORSE THAN BASELINE!")
    
    better_than_baseline = all(acc > baseline_accuracy for acc in model_accuracies.values())
    
    return better_than_baseline


def main():
    print("=" * 60)
    print("PIPELINE DIAGNOSTIC CHECK")
    print("=" * 60)
    print("\nExpected: 92-93% accuracy")
    print("Actual:   49% accuracy (random chance)")
    print("\nInvestigating root cause...")
    
    # Load cached preprocessed data (try v2 first, then v1)
    cache_file_v2 = project_root / "services" / "manage-agent" / "outputs" / "preprocessed_data_cache_v2.pkl"
    cache_file_v1 = project_root / "services" / "manage-agent" / "outputs" / "preprocessed_data_cache.pkl"
    
    if cache_file_v2.exists():
        cache_file = cache_file_v2
        print(f"\nUsing FIXED cache: {cache_file}")
    elif cache_file_v1.exists():
        cache_file = cache_file_v1
        print(f"\nUsing original cache: {cache_file}")
    else:
        print(f"\n❌ ERROR: Cache file not found")
        print(f"  Tried: {cache_file_v2}")
        print(f"  Tried: {cache_file_v1}")
        return
    
    print(f"\nLoading data from: {cache_file}")
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['train']
    X_val = data['val']
    X_test = data['test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    # Check if ESI levels are already mapped
    # If they contain 7.0, they need mapping
    if 7.0 in y_train.unique() or 7 in y_train.unique():
        print("\n⚠️  ESI levels not mapped yet. Applying mapping...")
        forward_mapping = {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, 4.0: 4, 5.0: 5, 7.0: 6}
        y_train = pd.Series([forward_mapping[float(v)] for v in y_train], dtype=np.int32)
        y_val = pd.Series([forward_mapping[float(v)] for v in y_val], dtype=np.int32)
        y_test = pd.Series([forward_mapping[float(v)] for v in y_test], dtype=np.int32)
    
    # Run all checks
    results = {}
    
    results['data_sanity'] = check_data_sanity(X_train, X_val, X_test, y_train, y_val, y_test)
    results['target_encoding'] = check_target_encoding(y_train, y_val, y_test)
    results['class_distribution'] = check_class_distribution(y_train, y_val, y_test)
    results['smote_leakage'] = check_smote_leakage(X_train, X_val, X_test, y_train, y_val, y_test)
    results['preprocessing_scale'] = check_preprocessing_scale(X_train, X_val, X_test)
    
    # Model accuracies (from previous runs)
    model_accuracies = {
        'Logistic Regression': 0.2339,
        'Random Forest': 0.5055,
        'XGBoost': 0.4937
    }
    results['baseline_comparison'] = check_baseline_comparison(y_val, model_accuracies)
    
    # Final diagnosis
    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)
    
    print("\nCheck Results:")
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check_name:25s}: {status}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDED FIXES")
    print("=" * 60)
    
    if not results['target_encoding']:
        print("\n❌ ISSUE: Target encoding incorrect")
        print("   Fix: Ensure ESI 7.0 → 6.0 mapping is applied BEFORE training")
    
    if not results['class_distribution']:
        print("\n❌ ISSUE: Class distribution problem")
        if not results['smote_leakage']:
            print("   Fix: SMOTE may have been applied incorrectly")
            print("   Check: Preprocessing pipeline should apply SMOTE ONLY to training set")
    
    if not results['smote_leakage']:
        print("\n❌ CRITICAL: SMOTE leakage detected!")
        print("   Fix: SMOTE was likely applied before train/val/test split")
        print("   Action: Re-run preprocessing with SMOTE applied ONLY to training set")
    
    if not results['preprocessing_scale']:
        print("\n❌ ISSUE: Feature scaling mismatch")
        print("   Fix: StandardScaler should be fitted on train, then transform val/test")
    
    if not results['baseline_comparison']:
        print("\n❌ ISSUE: Models performing worse than baseline")
        print("   Fix: Data pipeline has fundamental issues - check all above")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

