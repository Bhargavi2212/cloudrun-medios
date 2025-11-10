"""
Train baseline models for ESI triage classification.

Trains Logistic Regression and Random Forest as baseline models
to validate preprocessing and establish performance floor.
"""

import sys
from pathlib import Path
import pickle
import time
import json
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)
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
    # Forward mapping: original ESI → sequential
    forward_mapping = {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, 4.0: 4, 5.0: 5, 7.0: 6}
    
    # Inverse mapping: sequential → original ESI
    inverse_mapping = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0, 6: 7.0}
    
    # Apply mapping
    y_mapped = np.array([forward_mapping[float(val)] for val in y], dtype=np.int32)
    
    return y_mapped, forward_mapping, inverse_mapping


def train_logistic_regression(X_train, y_train, X_val, y_val):
    """Train Logistic Regression with L2 regularization."""
    print("\n[Model 1] Training Logistic Regression (L2 Regularization)...")
    start_time = time.time()
    
    # Convert to numpy if needed
    if isinstance(X_train, pd.DataFrame):
        X_train_array = X_train.values
        y_train_array = y_train.values
    else:
        X_train_array = X_train
        y_train_array = y_train
    
    if isinstance(X_val, pd.DataFrame):
        X_val_array = X_val.values
    else:
        X_val_array = X_val
    
    print(f"  Training samples: {len(X_train_array):,}")
    print(f"  Features: {X_train_array.shape[1]}")
    
    unique_classes = np.unique(y_train_array)
    print(f"  Classes to predict: {unique_classes}")
    
    print(f"  Using LogisticRegression with L2 regularization")
    
    # Use LogisticRegression (not SGDClassifier)
    lr = LogisticRegression(
        penalty='l2',                    # L2 (Ridge) regularization
        C=1.0,                           # Inverse regularization strength (1/alpha)
        max_iter=1000,                   # Maximum iterations
        solver='lbfgs',                  # L-BFGS solver (good for small-medium datasets)
        multi_class='multinomial',       # Multi-class classification
        class_weight='balanced',          # Handle class imbalance
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    print(f"  Training on full dataset...")
    lr.fit(X_train_array, y_train_array)
    
    print(f"  Training complete!")
    training_time = time.time() - start_time
    
    # Evaluate
    y_pred = lr.predict(X_val_array)
    acc = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average='macro')
    f1_weighted = f1_score(y_val, y_pred, average='weighted')
    
    print(f"  Training time: {training_time:.1f} seconds")
    print(f"  Validation Accuracy: {acc:.4f}")
    print(f"  Macro F1: {f1_macro:.4f}")
    print(f"  Weighted F1: {f1_weighted:.4f}")
    
    return lr, {
        'training_time': training_time,
        'accuracy': float(acc),
        'macro_f1': float(f1_macro),
        'weighted_f1': float(f1_weighted),
        'predictions': y_pred.tolist()
    }


def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest baseline model."""
    print("\n[Model 2] Training Random Forest...")
    start_time = time.time()
    
    # Train model
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    rf.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Evaluate
    y_pred = rf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average='macro')
    f1_weighted = f1_score(y_val, y_pred, average='weighted')
    
    print(f"  Training time: {training_time:.1f} seconds")
    print(f"  Validation Accuracy: {acc:.4f}")
    print(f"  Macro F1: {f1_macro:.4f}")
    print(f"  Weighted F1: {f1_weighted:.4f}")
    
    return rf, {
        'training_time': training_time,
        'accuracy': float(acc),
        'macro_f1': float(f1_macro),
        'weighted_f1': float(f1_weighted),
        'predictions': y_pred.tolist()
    }


def main():
    print("=" * 60)
    print("BASELINE MODEL TRAINING")
    print("=" * 60)
    
    # Load cached preprocessed data (try v9 first, then v8, v7, v4, v3, v2, v1)
    cache_v9_nlp_5class = project_root / "services" / "manage-agent" / "outputs" / "preprocessed_data_cache_v9_nlp_5class.pkl"
    cache_v8_rfv_clustered = project_root / "services" / "manage-agent" / "outputs" / "preprocessed_data_cache_v8_rfv_clustered.pkl"
    cache_v7_minimal = project_root / "services" / "manage-agent" / "outputs" / "preprocessed_data_cache_v7_minimal.pkl"
    cache_v4 = project_root / "services" / "manage-agent" / "outputs" / "preprocessed_data_cache_v4.pkl"
    cache_v3 = project_root / "services" / "manage-agent" / "outputs" / "preprocessed_data_cache_v3.pkl"
    cache_v2 = project_root / "services" / "manage-agent" / "outputs" / "preprocessed_data_cache_v2.pkl"
    cache_v1 = project_root / "services" / "manage-agent" / "outputs" / "preprocessed_data_cache.pkl"
    
    if cache_v9_nlp_5class.exists():
        cache_file = cache_v9_nlp_5class
        print(f"Using v9_nlp_5class cache (NLP embeddings + 5-class severity)")
    elif cache_v8_rfv_clustered.exists():
        cache_file = cache_v8_rfv_clustered
        print(f"Using v8_rfv_clustered cache (RFV clustering: 723 codes → 10-15 clusters)")
    elif cache_v7_minimal.exists():
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
    
    # Handle different cache formats (v8 uses 'X_train', older versions use 'train')
    if 'X_train' in data:
        X_train = data['X_train']
        X_val = data['X_val']
        X_test = data['X_test']
        y_train = data['y_train']
        y_val = data['y_val']
        y_test = data['y_test']
    else:
        X_train = data['train']
        X_val = data['val']
        X_test = data['test']
        y_train = data['y_train']
        y_val = data['y_val']
        y_test = data['y_test']
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Features: {len(X_train.columns)}")
    
    # Check ESI levels and handle different formats
    print(f"\nChecking ESI levels: {sorted(y_train.unique())}")
    
    # Check if using 5-class severity (v9 cache)
    unique_levels = sorted(y_train.unique())
    using_5class = False
    if set(unique_levels).issubset({1, 2, 3, 4, 5}) and len(unique_levels) == 5:
        # 5-class severity (v9 cache) - already mapped
        using_5class = True
        print("Using 5-class severity labels (1-5), already mapped from 7-class ESI")
        y_train_mapped = y_train.astype(np.int32)
        y_val_mapped = y_val.astype(np.int32)
        y_test_mapped = y_test.astype(np.int32)
        # Create mappings for metadata (5-class severity)
        forward_mapping = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        inverse_mapping = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        severity_labels = {1: "Critical", 2: "Emergent", 3: "Urgent", 4: "Standard", 5: "Non-urgent"}
    elif 7.0 in y_train.unique() or 7 in y_train.unique():
        # Need to map: [0,1,2,3,4,5,7] → [0,1,2,3,4,5,6]
        print("Mapping ESI levels: [0,1,2,3,4,5,7] → [0,1,2,3,4,5,6]")
        y_train_mapped, forward_mapping, inverse_mapping = map_esi_levels(y_train)
        y_val_mapped, _, _ = map_esi_levels(y_val)
        y_test_mapped, _, _ = map_esi_levels(y_test)
    else:
        # Already encoded (v2 cache) - 6 classes [0,1,2,3,4,5,6]
        print("ESI levels already encoded (using v2 cache)")
        y_train_mapped = y_train.astype(np.int32)
        y_val_mapped = y_val.astype(np.int32)
        y_test_mapped = y_test.astype(np.int32)
        # Create mappings for metadata
        forward_mapping = {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, 4.0: 4, 5.0: 5, 7.0: 6}
        inverse_mapping = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0, 6: 7.0}
    
    print(f"Final target levels: {sorted(np.unique(y_train_mapped))}")
    
    # Train models
    lr_model, lr_results = train_logistic_regression(
        X_train.values, y_train_mapped, X_val.values, y_val_mapped
    )
    
    rf_model, rf_results = train_random_forest(
        X_train.values, y_train_mapped, X_val.values, y_val_mapped
    )
    
    # Generate classification reports
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORTS")
    print("=" * 60)
    
    # Generate target names based on class count
    if using_5class:
        target_names = [f'Severity {i} ({severity_labels[i]})' for i in sorted(unique_levels)]
        n_classes = 5
    else:
        target_names = [f'ESI {inverse_mapping[i]}' for i in range(7)]
        n_classes = 7
    
    print("\nLogistic Regression:")
    print(classification_report(
        y_val_mapped, lr_results['predictions'],
        target_names=target_names
    ))
    
    print("\nRandom Forest:")
    print(classification_report(
        y_val_mapped, rf_results['predictions'],
        target_names=target_names
    ))
    
    # Save models and results
    models_dir = project_root / "services" / "manage-agent" / "models"
    models_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 60)
    print("SAVING MODELS")
    print("=" * 60)
    
    # Save Logistic Regression
    lr_path = models_dir / "logistic_regression.pkl"
    joblib.dump(lr_model, lr_path)
    print(f"✓ Logistic Regression saved: {lr_path}")
    
    # Save Random Forest
    rf_path = models_dir / "random_forest.pkl"
    joblib.dump(rf_model, rf_path)
    print(f"✓ Random Forest saved: {rf_path}")
    
    # Save metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'esi_mapping': {
            'forward': {str(k): v for k, v in forward_mapping.items()},
            'inverse': {str(k): v for k, v in inverse_mapping.items()}
        },
        'feature_names': list(X_train.columns),
        'n_features': len(X_train.columns),
        'models': {
            'logistic_regression': {
                'hyperparameters': lr_model.get_params(),
                **lr_results
            },
            'random_forest': {
                'hyperparameters': rf_model.get_params(),
                **rf_results
            }
        }
    }
    
    metadata_path = models_dir / "baseline_metadata.pkl"
    joblib.dump(metadata, metadata_path)
    print(f"✓ Metadata saved: {metadata_path}")
    
    # Save results JSON
    results = {
        'logistic_regression': {
            'training_time_seconds': lr_results['training_time'],
            'accuracy': lr_results['accuracy'],
            'macro_f1': lr_results['macro_f1'],
            'weighted_f1': lr_results['weighted_f1']
        },
        'random_forest': {
            'training_time_seconds': rf_results['training_time'],
            'accuracy': rf_results['accuracy'],
            'macro_f1': rf_results['macro_f1'],
            'weighted_f1': rf_results['weighted_f1']
        }
    }
    
    results_path = project_root / "services" / "manage-agent" / "outputs" / "baseline_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved: {results_path}")
    
    print("\n" + "=" * 60)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 60)
    print(f"Logistic Regression: {lr_results['accuracy']:.4f} accuracy, {lr_results['macro_f1']:.4f} macro F1")
    print(f"Random Forest: {rf_results['accuracy']:.4f} accuracy, {rf_results['macro_f1']:.4f} macro F1")
    print("\n✓ Preprocessing validated. Baseline models ready.")


if __name__ == "__main__":
    main()

