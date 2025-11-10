"""
Train XGBoost and LightGBM models with v9 cache (NLP embeddings + 5-class severity).

Compares both gradient boosting models for ESI triage classification.
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
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import joblib
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'services' / 'manage-agent'))


def calculate_sample_weights_5class(y_train):
    """
    Calculate sample weights for 5-class severity training.
    
    Uses 0-based severity labels (0-4):
    - Severity 0 (Critical, originally 1): 50x weight
    - Severity 1 (Emergent, originally 2): 10x weight
    - Others: 1x weight
    
    Args:
        y_train: Severity labels (0-4, after remapping)
        
    Returns:
        Array of sample weights
    """
    weights = np.ones(len(y_train), dtype=np.float32)
    
    for i, severity in enumerate(y_train):
        if severity == 0:  # Critical (originally 1) - highest priority
            weights[i] = 50.0
        elif severity == 1:  # Emergent (originally 2)
            weights[i] = 10.0
        else:
            weights[i] = 1.0
    
    return weights


def train_xgboost(X_train, y_train, X_val, y_val, feature_names):
    """Train XGBoost model."""
    print("\n" + "=" * 60)
    print("TRAINING XGBOOST")
    print("=" * 60)
    
    print("\n[Step 1] Configuring XGBoost...")
    
    # Research-based hyperparameters
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
    
    # Calculate sample weights
    print("[Step 2] Calculating sample weights (Severity 1: 50x, Severity 2: 10x)...")
    sample_weights = calculate_sample_weights_5class(y_train)
    
    print(f"  Sample weight distribution:")
    print(f"    Weight 1.0: {(sample_weights == 1.0).sum():,} samples")
    print(f"    Weight 10.0: {(sample_weights == 10.0).sum():,} samples")
    print(f"    Weight 50.0: {(sample_weights == 50.0).sum():,} samples")
    
    # Train
    print("[Step 3] Training XGBoost with early stopping...")
    start_time = time.time()
    
    xgb.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=10
    )
    
    training_time = time.time() - start_time
    
    print(f"\n[Step 4] Training complete (took {training_time:.1f} seconds)")
    if hasattr(xgb, 'best_iteration'):
        print(f"  Best iteration: {xgb.best_iteration}")
        print(f"  Best score: {xgb.best_score:.4f}")
    
    return xgb, training_time


def train_lightgbm(X_train, y_train, X_val, y_val, feature_names):
    """Train LightGBM model."""
    print("\n" + "=" * 60)
    print("TRAINING LIGHTGBM")
    print("=" * 60)
    
    print("\n[Step 1] Configuring LightGBM...")
    
    # LightGBM hyperparameters (similar to XGBoost)
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
    
    # Calculate sample weights
    print("[Step 2] Calculating sample weights (Severity 1: 50x, Severity 2: 10x)...")
    sample_weights = calculate_sample_weights_5class(y_train)
    
    print(f"  Sample weight distribution:")
    print(f"    Weight 1.0: {(sample_weights == 1.0).sum():,} samples")
    print(f"    Weight 10.0: {(sample_weights == 10.0).sum():,} samples")
    print(f"    Weight 50.0: {(sample_weights == 50.0).sum():,} samples")
    
    # Train
    print("[Step 3] Training LightGBM with early stopping...")
    start_time = time.time()
    
    lgbm.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(15, verbose=False)]
    )
    
    training_time = time.time() - start_time
    
    print(f"\n[Step 4] Training complete (took {training_time:.1f} seconds)")
    if hasattr(lgbm, 'best_iteration_'):
        print(f"  Best iteration: {lgbm.best_iteration_}")
    
    return lgbm, training_time


def evaluate_model(model, X_val, y_val, model_name, severity_labels):
    """Evaluate model on validation set."""
    print(f"\n[Evaluation] Evaluating {model_name}...")
    
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)
    
    # Metrics
    acc = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average='macro')
    f1_weighted = f1_score(y_val, y_pred, average='weighted')
    
    # AUROC (multiclass, one-vs-rest)
    n_classes = len(severity_labels)
    y_val_binarized = label_binarize(y_val, classes=list(range(n_classes)))
    auc_ovr = roc_auc_score(y_val_binarized, y_pred_proba, average='macro', multi_class='ovr')
    
    # Per-class metrics (0-based labels, but show as 1-5 for clarity)
    target_names = [f'Severity {i+1} ({severity_labels[i]})' for i in sorted(severity_labels.keys())]
    class_report = classification_report(
        y_val, y_pred,
        target_names=target_names,
        output_dict=True
    )
    
    # Severity 0 (Critical, originally 1) recall
    severity_0_recall = class_report[f'Severity 1 ({severity_labels[0]})']['recall']
    
    print(f"\n{model_name.upper()} VALIDATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:       {acc:.4f}")
    print(f"Macro F1:       {f1_macro:.4f}")
    print(f"Weighted F1:    {f1_weighted:.4f}")
    print(f"AUROC (OvR):    {auc_ovr:.4f}")
    print(f"\nPer-Class Performance:")
    print(classification_report(y_val, y_pred, target_names=target_names))
    
    print(f"\nSeverity 1 (Critical) Recall: {severity_0_recall:.4f}", end=" ")
    if severity_0_recall >= 0.85:
        print("✓ (≥0.85 requirement met)")
    else:
        print("⚠ (≥0.85 requirement NOT met)")
    
    return {
        'accuracy': float(acc),
        'macro_f1': float(f1_macro),
        'weighted_f1': float(f1_weighted),
        'auroc': float(auc_ovr),
        'severity_1_recall': float(severity_0_recall),
        'class_report': class_report,
        'predictions': y_pred.tolist(),
        'predictions_proba': y_pred_proba.tolist()
    }


def get_feature_importance(model, feature_names, model_name):
    """Get top 10 feature importances."""
    importance = model.feature_importances_
    feature_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features ({model_name}):")
    for idx, row in feature_imp_df.head(10).iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:.4f}")
    
    return feature_imp_df.to_dict('records')


def main():
    print("=" * 80)
    print("XGBOOST & LIGHTGBM TRAINING (v9: NLP Embeddings + 5-Class Severity)")
    print("=" * 80)
    
    # Load v9 cache
    cache_v9 = project_root / "services" / "manage-agent" / "outputs" / "preprocessed_data_cache_v9_nlp_5class.pkl"
    
    if not cache_v9.exists():
        print(f"ERROR: v9 cache not found: {cache_v9}")
        print("Please run save_preprocessed_cache_v9_nlp_5class.py first")
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
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Features: {len(feature_names)}")
    
    # Check if using 5-class severity
    unique_levels = sorted(y_train.unique())
    severity_labels = {1: "Critical", 2: "Emergent", 3: "Urgent", 4: "Standard", 5: "Non-urgent"}
    
    if set(unique_levels).issubset({1, 2, 3, 4, 5}) and len(unique_levels) == 5:
        print(f"\nUsing 5-class severity labels: {unique_levels}")
        print("  Remapping to 0-based for XGBoost/LightGBM: 1→0, 2→1, 3→2, 4→3, 5→4")
        # Remap to 0-based: 1→0, 2→1, 3→2, 4→3, 5→4
        y_train = (y_train - 1).astype(np.int32)
        y_val = (y_val - 1).astype(np.int32)
        y_test = (y_test - 1).astype(np.int32)
        # Update severity labels for 0-based
        severity_labels_0based = {0: "Critical", 1: "Emergent", 2: "Urgent", 3: "Standard", 4: "Non-urgent"}
        severity_labels = severity_labels_0based
    else:
        print(f"ERROR: Expected 5-class severity (1-5), got: {unique_levels}")
        return
    
    # Convert to numpy arrays
    X_train_array = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    X_val_array = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
    X_test_array = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    
    # Train XGBoost
    xgb_model, xgb_time = train_xgboost(
        X_train_array, y_train, X_val_array, y_val, feature_names
    )
    
    # Train LightGBM
    lgbm_model, lgbm_time = train_lightgbm(
        X_train_array, y_train, X_val_array, y_val, feature_names
    )
    
    # Evaluate both models
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    xgb_results = evaluate_model(xgb_model, X_val_array, y_val, "XGBoost", severity_labels)
    lgbm_results = evaluate_model(lgbm_model, X_val_array, y_val, "LightGBM", severity_labels)
    
    # Feature importance
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE")
    print("=" * 80)
    
    xgb_importance = get_feature_importance(xgb_model, feature_names, "XGBoost")
    lgbm_importance = get_feature_importance(lgbm_model, feature_names, "LightGBM")
    
    # Comparison summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Metric':<20} {'XGBoost':<15} {'LightGBM':<15} {'Winner':<15}")
    print("-" * 65)
    
    metrics = [
        ('Accuracy', 'accuracy'),
        ('Macro F1', 'macro_f1'),
        ('Weighted F1', 'weighted_f1'),
        ('AUROC', 'auroc'),
        ('Severity 1 Recall (Critical)', 'severity_1_recall'),
    ]
    
    for metric_name, metric_key in metrics:
        xgb_val = xgb_results[metric_key]
        lgbm_val = lgbm_results[metric_key]
        winner = "XGBoost" if xgb_val > lgbm_val else "LightGBM" if lgbm_val > xgb_val else "Tie"
        print(f"{metric_name:<20} {xgb_val:<15.4f} {lgbm_val:<15.4f} {winner:<15}")
    
    print(f"\n{'Training Time':<20} {xgb_time:<15.1f} {lgbm_time:<15.1f} {'Faster':<15}")
    
    # Save models
    models_dir = project_root / "services" / "manage-agent" / "models"
    models_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 80)
    print("SAVING MODELS")
    print("=" * 80)
    
    # Save XGBoost
    xgb_path = models_dir / "xgboost_triage.pkl"
    joblib.dump(xgb_model, xgb_path)
    print(f"✓ XGBoost saved: {xgb_path}")
    
    # Save LightGBM
    lgbm_path = models_dir / "lightgbm_triage.pkl"
    joblib.dump(lgbm_model, lgbm_path)
    print(f"✓ LightGBM saved: {lgbm_path}")
    
    # Save metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'cache_version': 'v9_nlp_5class',
        'classes': 5,
        'severity_labels': {i+1: severity_labels[i] for i in range(5)},  # Show as 1-5
        'severity_labels_0based': severity_labels,  # Actual 0-based labels used
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'xgboost': {
            'training_time': xgb_time,
            **xgb_results,
            'feature_importance': xgb_importance[:10]  # Top 10
        },
        'lightgbm': {
            'training_time': lgbm_time,
            **lgbm_results,
            'feature_importance': lgbm_importance[:10]  # Top 10
        }
    }
    
    metadata_path = models_dir / "xgboost_lightgbm_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✓ Metadata saved: {metadata_path}")
    
    # Save results as JSON
    results_path = project_root / "services" / "manage-agent" / "outputs" / "xgboost_lightgbm_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'xgboost': {k: v for k, v in xgb_results.items() if k != 'class_report' and k != 'predictions' and k != 'predictions_proba'},
            'lightgbm': {k: v for k, v in lgbm_results.items() if k != 'class_report' and k != 'predictions' and k != 'predictions_proba'},
            'comparison': {
                'best_accuracy': 'XGBoost' if xgb_results['accuracy'] > lgbm_results['accuracy'] else 'LightGBM',
                'best_f1': 'XGBoost' if xgb_results['macro_f1'] > lgbm_results['macro_f1'] else 'LightGBM',
                'best_auroc': 'XGBoost' if xgb_results['auroc'] > lgbm_results['auroc'] else 'LightGBM',
                'fastest': 'XGBoost' if xgb_time < lgbm_time else 'LightGBM'
            }
        }, f, indent=2)
    print(f"✓ Results saved: {results_path}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

