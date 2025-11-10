"""
Train XGBoost model for ESI triage classification.

Uses research-validated hyperparameters from Thailand 2025 study
with custom sample weights for critical ESI 1 cases.
"""

import sys
from pathlib import Path
import pickle
import time
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'services' / 'manage-agent'))


def map_esi_levels(y):
    """
    Map ESI levels from [0,1,2,3,4,5,7] to sequential [0,1,2,3,4,5,6].
    
    Args:
        y: Series or array of ESI levels (float)
        
    Returns:
        Mapped array (int32) and mapping dictionaries
    """
    forward_mapping = {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, 4.0: 4, 5.0: 5, 7.0: 6}
    inverse_mapping = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0, 6: 7.0}
    y_mapped = np.array([forward_mapping[float(val)] for val in y], dtype=np.int32)
    return y_mapped, forward_mapping, inverse_mapping


def calculate_sample_weights(y_train_mapped):
    """
    Calculate sample weights for XGBoost training (AGGRESSIVE).
    
    Uses mapped labels (0-6) where:
    - ESI 0 (resuscitation) → mapped 0: 10x weight
    - ESI 1 (critical) → mapped 1: 50x weight
    - ESI 2 (emergent) → mapped 2: 10x weight
    - Others: 1x weight
    
    Args:
        y_train_mapped: Mapped labels (0-6)
        
    Returns:
        Array of sample weights
    """
    weights = np.ones(len(y_train_mapped), dtype=np.float32)
    
    # Aggressive weighting using mapped labels
    for i, mapped_label in enumerate(y_train_mapped):
        if mapped_label == 1:  # ESI 1 (critical) - highest priority
            weights[i] = 50.0
        elif mapped_label == 2:  # ESI 2 (emergent)
            weights[i] = 10.0
        elif mapped_label == 0:  # ESI 0 (resuscitation)
            weights[i] = 10.0
        else:
            weights[i] = 1.0
    
    return weights


def train_xgboost(X_train, y_train, X_val, y_val, feature_names):
    """Train XGBoost model with research-based hyperparameters."""
    print("\n[Step 1] Configuring XGBoost with research hyperparameters...")
    
    # Research-based hyperparameters (Thailand 2025 study)
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=12,           # Deeper for complex patterns
        learning_rate=0.01,     # Slower learning for better convergence
        subsample=0.8,
        colsample_bytree=0.6,   # Feature subsampling
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
    
    # Calculate sample weights (AGGRESSIVE)
    print("[Step 2] Calculating sample weights (ESI 1: 50x, ESI 2: 10x, ESI 0: 10x)...")
    sample_weights = calculate_sample_weights(y_train)
    
    print(f"  Sample weight distribution:")
    print(f"    Weight 1.0: {(sample_weights == 1.0).sum():,} samples")
    print(f"    Weight 10.0: {(sample_weights == 10.0).sum():,} samples")
    print(f"    Weight 50.0: {(sample_weights == 50.0).sum():,} samples")
    
    # Prepare validation set for early stopping
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
    print(f"  Best iteration: {xgb.best_iteration}")
    print(f"  Best score: {xgb.best_score:.4f}")
    
    return xgb, training_time


def evaluate_model(xgb, X_val, y_val, y_val_original, inverse_mapping):
    """Evaluate XGBoost model on validation set."""
    print("\n[Step 5] Evaluating on validation set...")
    
    y_pred = xgb.predict(X_val)
    y_pred_proba = xgb.predict_proba(X_val)
    
    # Metrics
    acc = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average='macro')
    f1_weighted = f1_score(y_val, y_pred, average='weighted')
    
    # AUROC (multiclass, one-vs-rest, macro average)
    y_val_binarized = label_binarize(y_val, classes=list(range(7)))
    auc_ovr = roc_auc_score(y_val_binarized, y_pred_proba, average='macro', multi_class='ovr')
    
    # Per-class metrics
    class_report = classification_report(
        y_val, y_pred,
        target_names=[f'ESI {inverse_mapping[i]}' for i in range(7)],
        output_dict=True
    )
    
    # ESI 1 recall (critical - check original ESI 1.0, which maps to encoded 1)
    esi_1_recall = class_report['ESI 1.0']['recall']
    
    print(f"\nVALIDATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:       {acc:.4f}")
    print(f"Macro F1:       {f1_macro:.4f}")
    print(f"Weighted F1:    {f1_weighted:.4f}")
    print(f"AUROC (OvR):    {auc_ovr:.4f}")
    print(f"\nPer-Class Performance:")
    print(classification_report(
        y_val, y_pred,
        target_names=[f'ESI {inverse_mapping[i]}' for i in range(7)]
    ))
    
    # Check ESI 1 recall requirement
    print(f"\nESI 1 Recall: {esi_1_recall:.4f}", end=" ")
    if esi_1_recall >= 0.85:
        print("✓ (≥0.85 requirement met)")
    else:
        print("⚠ (≥0.85 requirement NOT met)")
    
    return {
        'accuracy': float(acc),
        'macro_f1': float(f1_macro),
        'weighted_f1': float(f1_weighted),
        'auroc': float(auc_ovr),
        'esi_1_recall': float(esi_1_recall),
        'class_report': class_report,
        'predictions': y_pred.tolist(),
        'predictions_proba': y_pred_proba.tolist()
    }


def plot_feature_importance(xgb, feature_names, output_path):
    """Plot and save feature importance."""
    print("\n[Step 6] Generating feature importance plot...")
    
    # Get feature importance
    importance = xgb.feature_importances_
    feature_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Top 10 features
    top_10 = feature_imp_df.head(10)
    
    print("\nTop 10 Most Important Features:")
    for idx, row in top_10.iterrows():
        print(f"  {row['feature']:25s}: {row['importance']:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_10, x='importance', y='feature', palette='viridis')
    plt.xlabel('Feature Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Top 10 Feature Importance (XGBoost)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Feature importance plot saved: {output_path}")
    
    return feature_imp_df.to_dict('records')


def evaluate_test_set(xgb, X_test, y_test, inverse_mapping):
    """Evaluate on test set (final unbiased performance)."""
    print("\n[Step 7] Evaluating on test set...")
    
    y_pred = xgb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    print(f"\nTEST SET PERFORMANCE:")
    print("=" * 60)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro F1:  {f1_macro:.4f}")
    
    return {
        'accuracy': float(acc),
        'macro_f1': float(f1_macro)
    }


def main():
    print("=" * 60)
    print("XGBOOST TRAINING")
    print("=" * 60)
    
    # Load cached preprocessed data (try v7_minimal first, then v4, v3, v2, v1)
    cache_v7_minimal = project_root / "services" / "manage-agent" / "outputs" / "preprocessed_data_cache_v7_minimal.pkl"
    cache_v4 = project_root / "services" / "manage-agent" / "outputs" / "preprocessed_data_cache_v4.pkl"
    cache_v3 = project_root / "services" / "manage-agent" / "outputs" / "preprocessed_data_cache_v3.pkl"
    cache_v2 = project_root / "services" / "manage-agent" / "outputs" / "preprocessed_data_cache_v2.pkl"
    cache_v1 = project_root / "services" / "manage-agent" / "outputs" / "preprocessed_data_cache.pkl"
    
    if cache_v7_minimal.exists():
        cache_file = cache_v7_minimal
        print(f"Using v7_minimal cache (top 10 RFV + essential features)")
    elif cache_v4.exists():
        cache_file = cache_v4
        print(f"Using v4 cache (RFV one-hot encoding fix)")
    elif cache_v3.exists():
        cache_file = cache_v3
        print(f"Using v3 cache (RFV fix applied)")
    elif cache_v2.exists():
        cache_file = cache_v2
        print(f"Using v2 cache (fallback)")
    elif cache_v1.exists():
        cache_file = cache_v1
        print(f"Using v1 cache (fallback)")
    else:
        raise FileNotFoundError(f"No cache files found")
    
    print(f"\nLoading preprocessed data from: {cache_file}")
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['train']
    X_val = data['val']
    X_test = data['test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Features: {len(X_train.columns)}")
    
    # Check if ESI levels are already encoded (v2 cache has them encoded)
    print(f"\nChecking ESI levels: {sorted(y_train.unique())}")
    if 7.0 in y_train.unique() or 7 in y_train.unique():
        # Need to map: [0,1,2,3,4,5,7] → [0,1,2,3,4,5,6]
        print("Mapping ESI levels: [0,1,2,3,4,5,7] → [0,1,2,3,4,5,6]")
        y_train_mapped, forward_mapping, inverse_mapping = map_esi_levels(y_train)
        y_val_mapped, _, _ = map_esi_levels(y_val)
        y_test_mapped, _, _ = map_esi_levels(y_test)
    else:
        # Already encoded (v2 cache)
        print("ESI levels already encoded (using v2 cache)")
        y_train_mapped = y_train.astype(np.int32)
        y_val_mapped = y_val.astype(np.int32)
        y_test_mapped = y_test.astype(np.int32)
        # Create mappings for metadata
        forward_mapping = {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, 4.0: 4, 5.0: 5, 7.0: 6}
        inverse_mapping = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0, 6: 7.0}
    
    print(f"Final ESI levels: {sorted(np.unique(y_train_mapped))}")
    
    # Train XGBoost
    xgb, training_time = train_xgboost(
        X_train.values, y_train_mapped,
        X_val.values, y_val_mapped,
        list(X_train.columns)
    )
    
    # Evaluate on validation set
    val_results = evaluate_model(
        xgb, X_val.values, y_val_mapped, y_val.values, inverse_mapping
    )
    
    # Feature importance
    outputs_dir = project_root / "services" / "manage-agent" / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    feature_imp_path = outputs_dir / "feature_importance_xgboost.png"
    feature_importance = plot_feature_importance(
        xgb, list(X_train.columns), feature_imp_path
    )
    
    # Evaluate on test set
    test_results = evaluate_test_set(
        xgb, X_test.values, y_test_mapped, inverse_mapping
    )
    
    # Save model and metadata
    models_dir = project_root / "services" / "manage-agent" / "models"
    models_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    
    # Save model
    model_path = models_dir / "xgboost_triage.pkl"
    joblib.dump(xgb, model_path)
    print(f"✓ Model saved: {model_path}")
    
    # Save metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'training_time_seconds': training_time,
        'best_iteration': int(xgb.best_iteration),
        'best_score': float(xgb.best_score),
        'esi_mapping': {
            'forward': {str(k): v for k, v in forward_mapping.items()},
            'inverse': {str(k): v for k, v in inverse_mapping.items()}
        },
        'feature_names': list(X_train.columns),
        'n_features': len(X_train.columns),
        'hyperparameters': xgb.get_params(),
        'feature_importance': feature_importance[:20],  # Top 20
        'validation_metrics': val_results,
        'test_metrics': test_results
    }
    
    metadata_path = models_dir / "xgboost_metadata.pkl"
    joblib.dump(metadata, metadata_path)
    print(f"✓ Metadata saved: {metadata_path}")
    
    print("\n" + "=" * 60)
    print("XGBOOST TRAINING COMPLETE")
    print("=" * 60)
    print(f"Validation Accuracy: {val_results['accuracy']:.4f}")
    print(f"Test Accuracy:       {test_results['accuracy']:.4f}")
    print(f"Macro F1:            {val_results['macro_f1']:.4f}")
    print(f"AUROC:               {val_results['auroc']:.4f}")
    print(f"ESI 1 Recall:        {val_results['esi_1_recall']:.4f}")
    print("\n✓ Model saved and ready for deployment.")


if __name__ == "__main__":
    main()

