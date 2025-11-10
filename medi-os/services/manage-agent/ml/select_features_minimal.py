"""
Minimal Feature Selection by Importance

Selects top 10 RFV features by importance + essential features:
- Top 10 RFV (by importance)
- All 7 vitals
- Age
- RFV_3D (2 features)
- ambulance_arrival
- Top 2-3 non-RFV features
"""

import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'services' / 'manage-agent'))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def print_section(title, char="="):
    """Print a formatted section header."""
    print("\n" + char * 80)
    print(f"{title}")
    print(char * 80)


def main():
    print("=" * 80)
    print("MINIMAL FEATURE SELECTION BY IMPORTANCE")
    print("=" * 80)
    print("\nSelecting features:")
    print("  - Top 10 RFV features (by importance)")
    print("  - All 7 vital signs: pulse, sbp, dbp, o2_sat, temp_c, respiration, pain")
    print("  - Age")
    print("  - RFV_3D: rfv1_3d, rfv2_3d")
    print("  - ambulance_arrival (currently #1 important)")
    print("  - Top 2-3 additional non-RFV features")
    
    # Load v4 cache
    print_section("STEP 1: LOADING V4 CACHE")
    
    cache_v4 = project_root / "services" / "manage-agent" / "outputs" / "preprocessed_data_cache_v4.pkl"
    
    if not cache_v4.exists():
        raise FileNotFoundError(f"Cache v4 not found: {cache_v4}")
    
    print(f"\nLoading: {cache_v4}")
    with open(cache_v4, 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['train']
    X_val = data['val']
    X_test = data['test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    print(f"\nOriginal features: {len(X_train.columns)}")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Step 2: Train quick Random Forest to get importances
    print_section("STEP 2: GETTING FEATURE IMPORTANCES")
    
    print("\nTraining Random Forest for feature importance analysis...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    rf.fit(X_train.values, y_train.values)
    
    # Get feature importances
    importances = rf.feature_importances_
    feature_names = list(X_train.columns)
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 20 Most Important Features:")
    print(importance_df.head(20).to_string(index=False))
    
    # Step 3: Select features
    print_section("STEP 3: SELECTING MINIMAL FEATURES")
    
    selected_features = []
    
    # Always include: Age
    if 'age' in X_train.columns:
        selected_features.append('age')
        print(f"\n✓ Age: included")
    
    # Always include: All 7 vital signs
    vital_signs = ['pulse', 'sbp', 'dbp', 'o2_sat', 'temp_c', 'respiration', 'pain']
    available_vitals = [v for v in vital_signs if v in X_train.columns]
    selected_features.extend(available_vitals)
    print(f"✓ Vital signs: {len(available_vitals)} included ({available_vitals})")
    
    # Always include: RFV_3D features
    rfv_3d_features = ['rfv1_3d', 'rfv2_3d']
    available_rfv_3d = [f for f in rfv_3d_features if f in X_train.columns]
    selected_features.extend(available_rfv_3d)
    print(f"✓ RFV_3D: {len(available_rfv_3d)} included ({available_rfv_3d})")
    
    # Always include: ambulance_arrival
    if 'ambulance_arrival' in X_train.columns:
        selected_features.append('ambulance_arrival')
        print(f"✓ ambulance_arrival: included")
    
    # Select top 10 RFV features by importance
    rfv_features = [f for f in feature_names if (f.startswith('rfv1_') or f.startswith('rfv2_')) and not f.endswith('_3d')]
    rfv_importance = importance_df[importance_df['feature'].isin(rfv_features)].copy()
    top_10_rfv = rfv_importance.head(10)['feature'].tolist()
    selected_features.extend(top_10_rfv)
    print(f"\n✓ Top 10 RFV features (by importance):")
    for i, rfv in enumerate(top_10_rfv, 1):
        imp = rfv_importance[rfv_importance['feature'] == rfv]['importance'].values[0]
        print(f"    {i:2d}. {rfv:25s}: {imp:.4f}")
    
    # Select top 2-3 non-RFV features (excluding already selected)
    already_selected = set(selected_features)
    non_rfv_features = [f for f in feature_names 
                        if not f.startswith('rfv') 
                        and f not in already_selected
                        and f not in ['age', 'pulse', 'sbp', 'dbp', 'o2_sat', 'temp_c', 'respiration', 'pain', 'ambulance_arrival']]
    
    non_rfv_importance = importance_df[importance_df['feature'].isin(non_rfv_features)].copy()
    top_3_non_rfv = non_rfv_importance.head(3)['feature'].tolist()
    selected_features.extend(top_3_non_rfv)
    print(f"\n✓ Top 3 non-RFV features (by importance):")
    for i, feat in enumerate(top_3_non_rfv, 1):
        imp = non_rfv_importance[non_rfv_importance['feature'] == feat]['importance'].values[0]
        print(f"    {i}. {feat:25s}: {imp:.4f}")
    
    # Remove duplicates and ensure all exist
    selected_features = sorted(list(set(selected_features)))
    selected_features = [f for f in selected_features if f in X_train.columns]
    
    print_section("STEP 4: FEATURE SELECTION SUMMARY")
    
    print(f"\nSelected features breakdown:")
    print(f"  Age: {1 if 'age' in selected_features else 0}")
    print(f"  Vitals: {len([f for f in selected_features if f in vital_signs])}")
    print(f"  RFV one-hot: {len([f for f in selected_features if (f.startswith('rfv1_') or f.startswith('rfv2_')) and not f.endswith('_3d')])}")
    print(f"  RFV_3D: {len([f for f in selected_features if f.endswith('_3d')])}")
    print(f"  ambulance_arrival: {1 if 'ambulance_arrival' in selected_features else 0}")
    print(f"  Other non-RFV: {len([f for f in selected_features if f not in vital_signs and 'rfv' not in f.lower() and f != 'age' and f != 'ambulance_arrival'])}")
    print(f"\n  Total: {len(selected_features)} features (down from {len(X_train.columns)})")
    
    # Create new DataFrames with selected features
    print_section("STEP 5: CREATING MINIMAL FEATURE DATASETS")
    
    X_train_minimal = X_train[selected_features].copy()
    X_val_minimal = X_val[selected_features].copy()
    X_test_minimal = X_test[selected_features].copy()
    
    print(f"\nMinimal feature sets created:")
    print(f"  Train: {X_train_minimal.shape}")
    print(f"  Val:   {X_val_minimal.shape}")
    print(f"  Test:  {X_test_minimal.shape}")
    
    # Quick validation: Train on minimal features
    print_section("STEP 6: VALIDATION - TRAINING ON MINIMAL FEATURES")
    
    print("\nTraining Random Forest on minimal features for validation...")
    rf_minimal = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    rf_minimal.fit(X_train_minimal.values, y_train.values)
    acc_minimal = accuracy_score(y_val, rf_minimal.predict(X_val_minimal.values))
    
    print(f"  Validation accuracy with minimal features: {acc_minimal:.4f}")
    
    # Compare with full feature set
    acc_full = accuracy_score(y_val, rf.predict(X_val.values))
    print(f"  Validation accuracy with full features: {acc_full:.4f}")
    
    if acc_minimal >= acc_full * 0.95:  # Within 5% of full features
        print(f"  ✓ Minimal features perform well (within 5% of full)")
    else:
        print(f"  ⚠️  Minimal features show some accuracy drop")
    
    # Save minimal cache
    print_section("STEP 7: SAVING MINIMAL FEATURE CACHE")
    
    outputs_dir = project_root / "services" / "manage-agent" / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    cache_file = outputs_dir / "preprocessed_data_cache_v7_minimal.pkl"
    
    preprocessed_data = {
        'train': X_train_minimal,
        'val': X_val_minimal,
        'test': X_test_minimal,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'selected_features': selected_features,
        'importance_df': importance_df
    }
    
    print(f"\nSaving to: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(preprocessed_data, f)
    
    file_size = cache_file.stat().st_size / (1024 * 1024)  # MB
    print(f"✓ Cache saved! Size: {file_size:.2f} MB")
    
    # Save importance report
    importance_path = outputs_dir / "feature_importance_minimal_selection.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"✓ Importance report saved: {importance_path}")
    
    print_section("MINIMAL FEATURE SELECTION COMPLETE")
    
    print(f"\nSummary:")
    print(f"  Original features: {len(X_train.columns)}")
    print(f"  Selected features: {len(selected_features)}")
    print(f"  Reduction: {len(X_train.columns) - len(selected_features)} features removed")
    print(f"  Validation accuracy: {acc_minimal:.4f} (minimal) vs {acc_full:.4f} (full)")
    print(f"\nNext steps:")
    print(f"  1. Update training scripts to use v7_minimal cache")
    print(f"  2. Retrain models on minimal feature set")
    print(f"  3. Compare accuracy with full feature set")


if __name__ == "__main__":
    main()

