"""
Check features after preprocessing.

CRITICAL: Verify that important features (especially RFV1) are present
and not accidentally dropped during preprocessing.
"""

import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'services' / 'manage-agent'))


def check_critical_features(feature_names):
    """
    Check for critical features that must be present.
    
    Returns:
        dict with feature status and missing features
    """
    critical_features = {
        'RFV1 (chief complaint)': ['rfv1'],
        'RFV1 3D (chief complaint 3-digit)': ['rfv1_3d'],
        'Age': ['age'],
        'Vitals - Pulse': ['pulse'],
        'Vitals - SBP': ['sbp'],
        'Vitals - DBP': ['dbp'],
        'Vitals - O2 Sat': ['o2_sat'],
        'Vitals - Temp': ['temp_c'],
        'Vitals - Respiration': ['respiration'],
        'Pain Score': ['pain'],
        'Sex': ['sex'],
        'Wait Time': ['wait_time'],
        'Length of Visit': ['length_of_visit']
    }
    
    status = {}
    missing_critical = []
    
    for feature_group, possible_names in critical_features.items():
        found = False
        found_name = None
        
        for name in possible_names:
            if name in feature_names:
                found = True
                found_name = name
                break
        
        if found:
            status[feature_group] = f"✅ Present: {found_name}"
        else:
            status[feature_group] = f"❌ MISSING: {possible_names}"
            missing_critical.append(feature_group)
    
    return status, missing_critical


def check_feature_importance(model_path, feature_names):
    """Check feature importance from trained XGBoost model."""
    if not model_path.exists():
        print(f"\n⚠️  Model not found: {model_path}")
        print("   Run train_xgboost.py first to train the model")
        return None, None
    
    print(f"\nLoading XGBoost model: {model_path}")
    xgb = joblib.load(model_path)
    
    # Get feature importance
    importance = xgb.feature_importances_
    feature_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Top feature
    top_feature = feature_imp_df.iloc[0]
    top_importance = top_feature['importance']
    
    print(f"\nFeature Importance Analysis:")
    print(f"  Top feature: {top_feature['feature']} (importance: {top_importance:.4f})")
    print(f"  Top 10 features:")
    for idx, row in feature_imp_df.head(10).iterrows():
        print(f"    {row['feature']:25s}: {row['importance']:6.4f}")
    
    return top_importance, feature_imp_df


def main():
    print("=" * 60)
    print("FEATURE CHECK DIAGNOSTIC")
    print("=" * 60)
    
    # Load preprocessed data
    cache_file = project_root / "services" / "manage-agent" / "outputs" / "preprocessed_data_cache_v2.pkl"
    
    if not cache_file.exists():
        print(f"\n❌ Cache file not found: {cache_file}")
        return
    
    print(f"\nLoading preprocessed data from: {cache_file}")
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['train']
    feature_names = list(X_train.columns)
    
    print(f"\n" + "=" * 60)
    print("STEP 1: ALL FEATURES AFTER PREPROCESSING")
    print("=" * 60)
    print(f"\nTotal features: {len(feature_names)}")
    print(f"\nFeature list:")
    for i, feat in enumerate(feature_names, 1):
        print(f"  {i:2d}. {feat}")
    
    print(f"\n" + "=" * 60)
    print("STEP 2: CRITICAL FEATURE CHECK")
    print("=" * 60)
    
    status, missing_critical = check_critical_features(feature_names)
    
    print(f"\nCritical feature status:")
    for feature_group, status_msg in status.items():
        print(f"  {status_msg}")
    
    if missing_critical:
        print(f"\n❌ CRITICAL: {len(missing_critical)} critical features MISSING!")
        print(f"   Missing: {', '.join(missing_critical)}")
    else:
        print(f"\n✅ All critical features present")
    
    # Special check for RFV1
    print(f"\n" + "=" * 60)
    print("STEP 3: RFV1 CHECK (MOST IMPORTANT FEATURE)")
    print("=" * 60)
    
    rfv1_present = 'rfv1' in feature_names
    rfv1_3d_present = 'rfv1_3d' in feature_names
    
    if rfv1_present:
        print(f"✅ RFV1 (chief complaint) is present")
        print(f"   This is the MOST predictive feature - good!")
    else:
        print(f"❌ CRITICAL ERROR: RFV1 is MISSING!")
        print(f"   RFV1 is the chief complaint - without it, accuracy will be ~50%")
        print(f"   Action: Re-run preprocessing and ensure RFV1 is NOT dropped")
    
    if rfv1_3d_present:
        print(f"✅ RFV1_3D (3-digit code) is present")
    else:
        print(f"⚠️  RFV1_3D is missing (less critical than RFV1)")
    
    # Check vitals
    print(f"\n" + "=" * 60)
    print("STEP 4: VITALS CHECK")
    print("=" * 60)
    
    vitals = ['pulse', 'sbp', 'dbp', 'o2_sat', 'temp_c', 'respiration']
    vitals_present = [v for v in vitals if v in feature_names]
    vitals_missing = [v for v in vitals if v not in feature_names]
    
    print(f"\nVitals present: {len(vitals_present)}/{len(vitals)}")
    for vital in vitals_present:
        print(f"  ✅ {vital}")
    
    if vitals_missing:
        print(f"\nVitals missing: {len(vitals_missing)}")
        for vital in vitals_missing:
            print(f"  ❌ {vital}")
    
    if len(vitals_present) >= 5:
        print(f"\n✅ Sufficient vitals present ({len(vitals_present)}/6)")
    else:
        print(f"\n⚠️  Warning: Only {len(vitals_present)}/6 vitals present")
    
    # Check feature importance
    print(f"\n" + "=" * 60)
    print("STEP 5: FEATURE IMPORTANCE CHECK")
    print("=" * 60)
    
    model_path = project_root / "services" / "manage-agent" / "models" / "xgboost_triage.pkl"
    top_importance, feature_imp_df = check_feature_importance(model_path, feature_names)
    
    if top_importance is not None:
        print(f"\nTop feature importance: {top_importance:.4f}")
        
        if top_importance > 0.20:
            print(f"✅ Top feature importance > 0.20 (strong signal)")
        else:
            print(f"⚠️  Top feature importance < 0.20 (weak signal)")
            print(f"   This suggests features may not be very predictive")
        
        # Check if RFV1 is in top features
        if feature_imp_df is not None:
            rfv1_importance = feature_imp_df[feature_imp_df['feature'] == 'rfv1']
            if not rfv1_importance.empty:
                rfv1_imp = rfv1_importance.iloc[0]['importance']
                rfv1_rank = feature_imp_df[feature_imp_df['feature'] == 'rfv1'].index[0] + 1
                print(f"\nRFV1 importance: {rfv1_imp:.4f} (rank: {rfv1_rank})")
                if rfv1_rank <= 3:
                    print(f"✅ RFV1 is in top 3 most important features")
                else:
                    print(f"⚠️  RFV1 is not in top 3 (rank {rfv1_rank})")
    
    # Final diagnosis
    print(f"\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)
    
    issues = []
    
    if not rfv1_present:
        issues.append("CRITICAL: RFV1 (chief complaint) is MISSING")
        issues.append("  → This explains 50% accuracy (random chance)")
        issues.append("  → RFV1 is the most predictive feature")
        issues.append("  → Action: Re-preprocess WITHOUT dropping RFV1")
    
    if len(vitals_present) < 5:
        issues.append(f"Warning: Only {len(vitals_present)}/6 vitals present")
    
    if top_importance is not None and top_importance < 0.20:
        issues.append(f"Warning: Top feature importance ({top_importance:.4f}) < 0.20")
        issues.append("  → Features may have weak predictive signal")
    
    if missing_critical:
        issues.append(f"Critical features missing: {', '.join(missing_critical)}")
    
    if issues:
        print(f"\n❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print(f"\n✅ All checks passed!")
        print(f"   Features look good")
        print(f"   If accuracy is still low, the problem may be:")
        print(f"     - Inherently difficult classification task")
        print(f"     - Need more features or feature engineering")
        print(f"     - Need hyperparameter tuning or ensemble methods")
    
    print(f"\n" + "=" * 60)


if __name__ == "__main__":
    main()

