"""
Train models using Age + RFV + Vitals features.

When a patient enters, we have:
- Age
- RFV (Reason for Visit) - text description converted to embeddings
- Vitals (pulse, respiration, sbp, dbp, o2_sat, temp_c, pain, gcs, on_oxygen)

This tests if these basic features can predict ESI severity.
"""

import sys
from pathlib import Path
import pickle
import time
from datetime import datetime

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
import joblib
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'services' / 'manage-agent'))


def filter_age_rfv_vitals_features(X_train, X_val, X_test, feature_names):
    """
    Filter to Age + RFV + Vitals features.
    
    Includes:
    - Age (1 feature)
    - RFV features:
      * rfv1_emb_* (20 features) - RFV1 embeddings
      * rfv2_emb_* (20 features) - RFV2 embeddings
      * rfv1_3d, rfv2_3d (2 features) - RFV 3D features
    - Vitals (9 features):
      * pulse, respiration, sbp, dbp, o2_sat, temp_c, pain, gcs, on_oxygen
    """
    # Age feature
    age_features = ['age']
    
    # RFV embedding features
    rfv1_emb_features = [f for f in feature_names if f.startswith('rfv1_emb_')]
    rfv2_emb_features = [f for f in feature_names if f.startswith('rfv2_emb_')]
    
    # RFV 3D features
    rfv_3d_features = [f for f in feature_names if f in ['rfv1_3d', 'rfv2_3d']]
    
    # Combine all RFV features
    all_rfv_features = rfv1_emb_features + rfv2_emb_features + rfv_3d_features
    
    # Vital signs
    vital_features = ['pulse', 'respiration', 'sbp', 'dbp', 'o2_sat', 'temp_c', 'pain', 'gcs', 'on_oxygen']
    vital_features = [f for f in vital_features if f in feature_names]  # Only include if they exist
    
    # Final selected features
    selected_features = age_features + all_rfv_features + vital_features
    
    print(f"\n[Feature Selection] Filtering to Age + RFV + Vitals:")
    print(f"  Age: {len(age_features)} feature")
    print(f"  RFV1 embeddings: {len(rfv1_emb_features)} features")
    print(f"  RFV2 embeddings: {len(rfv2_emb_features)} features")
    print(f"  RFV 3D: {len(rfv_3d_features)} features")
    print(f"  Vitals: {len(vital_features)} features")
    print(f"    {', '.join(vital_features)}")
    print(f"  Total: {len(selected_features)} features (down from {len(feature_names)})")
    
    # Get indices of selected features
    feature_indices = [feature_names.index(f) for f in selected_features]
    
    # Filter DataFrames
    if isinstance(X_train, pd.DataFrame):
        X_train_filtered = X_train[selected_features]
        X_val_filtered = X_val[selected_features]
        X_test_filtered = X_test[selected_features]
    else:
        # Numpy arrays
        X_train_filtered = X_train[:, feature_indices]
        X_val_filtered = X_val[:, feature_indices]
        X_test_filtered = X_test[:, feature_indices]
    
    return X_train_filtered, X_val_filtered, X_test_filtered, selected_features


def calculate_sample_weights_5class(y_train):
    """Calculate sample weights for 5-class severity."""
    weights = np.ones(len(y_train), dtype=np.float32)
    for i, severity in enumerate(y_train):
        if severity == 0:  # Critical
            weights[i] = 50.0
        elif severity == 1:  # Emergent
            weights[i] = 10.0
        else:
            weights[i] = 1.0
    return weights


def train_model(model, model_name, X_train, y_train, X_val, y_val, sample_weights=None):
    """Train a model and return results."""
    print(f"\n[Training] {model_name}...")
    start_time = time.time()
    
    if sample_weights is not None and hasattr(model, 'fit'):
        # Models that support sample_weight
        if model_name in ['XGBoost', 'LightGBM']:
            if model_name == 'XGBoost':
                model.fit(X_train, y_train, sample_weight=sample_weights, 
                         eval_set=[(X_val, y_val)], verbose=False)
            else:  # LightGBM
                model.fit(X_train, y_train, sample_weight=sample_weights,
                         eval_set=[(X_val, y_val)], 
                         callbacks=[early_stopping(15, verbose=False)])
        else:
            model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Evaluate
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average='macro')
    f1_weighted = f1_score(y_val, y_pred, average='weighted')
    
    # AUROC
    y_pred_proba = model.predict_proba(X_val)
    n_classes = len(np.unique(y_val))
    y_val_binarized = label_binarize(y_val, classes=list(range(n_classes)))
    auc_ovr = roc_auc_score(y_val_binarized, y_pred_proba, average='macro', multi_class='ovr')
    
    # Severity 1 (Critical) recall
    severity_labels = {0: "Critical", 1: "Emergent", 2: "Urgent", 3: "Standard", 4: "Non-urgent"}
    target_names = [f'Severity {i+1} ({severity_labels[i]})' for i in sorted(severity_labels.keys())]
    class_report = classification_report(y_val, y_pred, target_names=target_names, output_dict=True)
    severity_1_recall = class_report[f'Severity 1 ({severity_labels[0]})']['recall']
    
    print(f"  Training time: {training_time:.1f}s")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro F1: {f1_macro:.4f}")
    print(f"  Severity 1 Recall: {severity_1_recall:.4f}")
    
    return {
        'model': model,
        'training_time': training_time,
        'accuracy': float(acc),
        'macro_f1': float(f1_macro),
        'weighted_f1': float(f1_weighted),
        'auroc': float(auc_ovr),
        'severity_1_recall': float(severity_1_recall),
        'class_report': class_report
    }


def main():
    print("=" * 80)
    print("TRIAGE: Age + RFV + Vitals")
    print("=" * 80)
    print("\nTesting if Age, RFV (Reason for Visit), and Vitals can predict ESI severity")
    print("This simulates triage when basic patient information is available.")
    
    # Load v9 cache
    cache_v9 = project_root / "services" / "manage-agent" / "outputs" / "preprocessed_data_cache_v9_nlp_5class.pkl"
    
    if not cache_v9.exists():
        print(f"ERROR: v9 cache not found: {cache_v9}")
        return
    
    print(f"\nLoading preprocessed data from: {cache_v9}")
    with open(cache_v9, 'rb') as f:
        data = pickle.load(f)
    
    # Handle different cache formats
    if 'X_train' in data:
        X_train = data['X_train']
        X_val = data['X_val']
        X_test = data['X_test']
        y_train = data['y_train']
        y_val = data['y_val']
        y_test = data['y_test']
        feature_names = data.get('feature_names', X_train.columns.tolist())
    else:
        X_train = data['train']
        X_val = data['val']
        X_test = data['test']
        y_train = data['y_train']
        y_val = data['y_val']
        y_test = data['y_test']
        feature_names = X_train.columns.tolist()
    
    print(f"Original features: {len(feature_names)}")
    
    # Filter to Age + RFV + Vitals
    X_train_filtered, X_val_filtered, X_test_filtered, selected_features = filter_age_rfv_vitals_features(
        X_train, X_val, X_test, feature_names
    )
    
    # Convert to numpy arrays
    X_train_array = X_train_filtered.values if isinstance(X_train_filtered, pd.DataFrame) else X_train_filtered
    X_val_array = X_val_filtered.values if isinstance(X_val_filtered, pd.DataFrame) else X_val_filtered
    X_test_array = X_test_filtered.values if isinstance(X_test_filtered, pd.DataFrame) else X_test_filtered
    
    # Handle 5-class severity labels
    unique_levels = sorted(y_train.unique())
    if set(unique_levels).issubset({1, 2, 3, 4, 5}) and len(unique_levels) == 5:
        print(f"\nUsing 5-class severity labels: {unique_levels}")
        print("  Remapping to 0-based: 1→0, 2→1, 3→2, 4→3, 5→4")
        y_train = (y_train - 1).astype(np.int32)
        y_val = (y_val - 1).astype(np.int32)
        y_test = (y_test - 1).astype(np.int32)
    
    # Calculate sample weights
    sample_weights = calculate_sample_weights_5class(y_train)
    
    print(f"\nTrain: {X_train_array.shape}, Val: {X_val_array.shape}, Test: {X_test_array.shape}")
    
    # Train multiple models
    print("\n" + "=" * 80)
    print("TRAINING MODELS (Age + RFV + Vitals)")
    print("=" * 80)
    
    results = {}
    
    # 1. Logistic Regression
    lr = LogisticRegression(
        penalty='l2',
        C=1.0,
        max_iter=1000,
        solver='lbfgs',
        multi_class='multinomial',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    results['LogisticRegression'] = train_model(
        lr, "Logistic Regression", 
        X_train_array, y_train, X_val_array, y_val
    )
    
    # 2. Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    results['RandomForest'] = train_model(
        rf, "Random Forest",
        X_train_array, y_train, X_val_array, y_val
    )
    
    # 3. XGBoost
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=12,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.6,
        gamma=1,
        min_child_weight=3,
        reg_alpha=0.5,
        reg_lambda=1.0,
        eval_metric='mlogloss',
        early_stopping_rounds=15,
        random_state=42,
        tree_method='hist',
        n_jobs=-1
    )
    results['XGBoost'] = train_model(
        xgb, "XGBoost",
        X_train_array, y_train, X_val_array, y_val, sample_weights
    )
    
    # 4. LightGBM
    lgbm = LGBMClassifier(
        n_estimators=300,
        max_depth=12,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_samples=3,
        reg_alpha=0.5,
        reg_lambda=1.0,
        objective='multiclass',
        metric='multi_logloss',
        early_stopping_rounds=15,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    results['LightGBM'] = train_model(
        lgbm, "LightGBM",
        X_train_array, y_train, X_val_array, y_val, sample_weights
    )
    
    # Comparison
    print("\n" + "=" * 80)
    print("MODEL COMPARISON (Age + RFV + Vitals)")
    print("=" * 80)
    
    print(f"\n{'Model':<20} {'Accuracy':<12} {'Macro F1':<12} {'Severity 1 Recall':<18} {'Training Time':<15}")
    print("-" * 80)
    
    for model_name, result in results.items():
        print(f"{model_name:<20} {result['accuracy']:<12.4f} {result['macro_f1']:<12.4f} "
              f"{result['severity_1_recall']:<18.4f} {result['training_time']:<15.1f}s")
    
    # Find best model
    best_accuracy = max(results.items(), key=lambda x: x[1]['accuracy'])
    best_f1 = max(results.items(), key=lambda x: x[1]['macro_f1'])
    
    print(f"\nBest Accuracy: {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.4f})")
    print(f"Best Macro F1: {best_f1[0]} ({best_f1[1]['macro_f1']:.4f})")
    
    # Save results
    output_dir = project_root / "services" / "manage-agent" / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    results_summary = {
        'features': 'Age + RFV + Vitals',
        'n_features': len(selected_features),
        'selected_features': selected_features,
        'models': {
            name: {
                'accuracy': result['accuracy'],
                'macro_f1': result['macro_f1'],
                'weighted_f1': result['weighted_f1'],
                'auroc': result['auroc'],
                'severity_1_recall': result['severity_1_recall'],
                'training_time': result['training_time']
            }
            for name, result in results.items()
        },
        'best_accuracy': best_accuracy[0],
        'best_f1': best_f1[0]
    }
    
    results_file = output_dir / "age_rfv_vitals_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print(f"\nWith Age + RFV + Vitals ({len(selected_features)} features):")
    print(f"  Can we predict ESI severity? {'YES' if best_accuracy[1]['accuracy'] > 0.3 else 'MARGINAL' if best_accuracy[1]['accuracy'] > 0.2 else 'NO'}")
    print(f"  Best model: {best_accuracy[0]} ({best_accuracy[1]['accuracy']*100:.1f}% accuracy)")
    print(f"  Critical recall: {best_accuracy[1]['severity_1_recall']*100:.1f}%")
    print("\nThis shows the essential information needed for triage!")


if __name__ == "__main__":
    main()

