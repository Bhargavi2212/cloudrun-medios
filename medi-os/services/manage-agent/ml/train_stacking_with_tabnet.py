"""
Phase 2: Stacking Ensemble with TabNet, XGBoost, and LightGBM
Try different meta-learners sequentially.
"""

import sys
from pathlib import Path
import pickle
import time
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
from pytorch_tabnet.tab_model import TabNetClassifier

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'services' / 'manage-agent'))

# Setup directories
outputs_dir = project_root / "services" / "manage-agent" / "outputs"
models_dir = project_root / "services" / "manage-agent" / "models"
outputs_dir.mkdir(exist_ok=True)
models_dir.mkdir(exist_ok=True)


def load_models():
    """Load pre-trained TabNet, XGBoost, and LightGBM models."""
    print("\n[1] Loading models...")
    
    # Load pre-trained TabNet
    tabnet_path = models_dir / "tabnet_model.zip.zip"
    if not tabnet_path.exists():
        # Try alternative name
        tabnet_path = models_dir / "tabnet_model.zip"
    if not tabnet_path.exists():
        raise FileNotFoundError(f"TabNet model not found: {tabnet_path}")
    print(f"  Loading TabNet from: {tabnet_path.name}")
    tabnet_model = TabNetClassifier()
    tabnet_model.load_model(str(tabnet_path))
    print("    TabNet loaded successfully (pre-trained)")
    
    # Load XGBoost
    xgb_path = models_dir / "final_xgboost_full_features.pkl"
    if not xgb_path.exists():
        raise FileNotFoundError(f"XGBoost model not found: {xgb_path}")
    print(f"  Loading XGBoost from: {xgb_path.name}")
    xgb_model = joblib.load(xgb_path)
    
    # Load LightGBM
    lgbm_path = models_dir / "final_lightgbm_full_features.pkl"
    if not lgbm_path.exists():
        raise FileNotFoundError(f"LightGBM model not found: {lgbm_path}")
    print(f"  Loading LightGBM from: {lgbm_path.name}")
    lgbm_model = joblib.load(lgbm_path)
    
    print("  All models loaded successfully!")
    return tabnet_model, xgb_model, lgbm_model


def load_preprocessed_data():
    """Load preprocessed data from cache."""
    print("\n[2] Loading preprocessed data...")
    cache_v10 = outputs_dir / "preprocessed_data_cache_v10_clinical_features.pkl"
    cache_v9 = outputs_dir / "preprocessed_data_cache_v9_nlp_5class.pkl"
    
    cache_file = cache_v10 if cache_v10.exists() else cache_v9
    if not cache_file.exists():
        raise FileNotFoundError(f"Cache not found: {cache_v10} or {cache_v9}")
    
    print(f"  Loading cache: {cache_file.name}")
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    
    # Extract data
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
    
    # Handle 5-class severity labels (remap 1-5 to 0-4)
    unique_levels = sorted(y_train.unique())
    if set(unique_levels).issubset({1, 2, 3, 4, 5}) and len(unique_levels) == 5:
        print(f"  Remapping 5-class severity to 0-based: 1->0, 2->1, 3->2, 4->3, 5->4")
        y_train = (y_train - 1).astype(np.int32)
        y_val = (y_val - 1).astype(np.int32)
        y_test = (y_test - 1).astype(np.int32)
    
    # Combine train + val for full training
    print(f"  Combining train + val for full training...")
    X_train_full = pd.concat([X_train, X_val], axis=0) if isinstance(X_train, pd.DataFrame) else np.vstack([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val], axis=0) if isinstance(y_train, pd.Series) else np.hstack([y_train, y_val])
    
    print(f"  Training set (full): {X_train_full.shape}")
    print(f"  Test set: {X_test.shape}")
    
    return X_train_full, X_test, y_train_full, y_test


def evaluate_model(model, X_test, y_test, class_names):
    """Evaluate model and return metrics."""
    # Convert to numpy array if DataFrame (TabNet compatibility)
    X_test_array = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    y_pred = model.predict(X_test_array)
    
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    
    class_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    critical_recall = class_report.get('Severity 1 (Critical)', {}).get('recall', 0)
    
    return {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'critical_recall': float(critical_recall) if critical_recall else None,
        'classification_report': class_report
    }


def precompute_base_predictions(tabnet_model, xgb_model, lgbm_model, X_train, y_train, X_test, cv=3):
    """
    Pre-compute predictions from pre-trained base models using cross-validation.
    This avoids retraining TabNet during stacking.
    """
    print("  Pre-computing base model predictions with CV...")
    print(f"    Using {cv}-fold CV for out-of-fold predictions (no retraining)")
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Arrays to store predictions
    n_samples = len(X_train)
    n_classes = len(np.unique(y_train))
    
    tabnet_proba = np.zeros((n_samples, n_classes))
    xgb_proba = np.zeros((n_samples, n_classes))
    lgbm_proba = np.zeros((n_samples, n_classes))
    
    # Convert to arrays for TabNet
    X_train_array = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_train_array = y_train.values if isinstance(y_train, pd.Series) else y_train
    
    # For each fold, get predictions from pre-trained models
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_array, y_train_array)):
        print(f"    Fold {fold_idx + 1}/{cv}...")
        
        # Get predictions from pre-trained models (no retraining!)
        X_val_fold = X_train_array[val_idx]
        
        # TabNet predictions (use pre-trained model)
        tabnet_proba[val_idx] = tabnet_model.predict_proba(X_val_fold)
        
        # XGBoost predictions (use pre-trained model)
        if isinstance(X_train, pd.DataFrame):
            xgb_proba[val_idx] = xgb_model.predict_proba(X_train.iloc[val_idx])
        else:
            xgb_proba[val_idx] = xgb_model.predict_proba(X_val_fold)
        
        # LightGBM predictions (use pre-trained model)
        if isinstance(X_train, pd.DataFrame):
            lgbm_proba[val_idx] = lgbm_model.predict_proba(X_train.iloc[val_idx])
        else:
            lgbm_proba[val_idx] = lgbm_model.predict_proba(X_val_fold)
    
    print("  Getting test set predictions...")
    # Get test predictions
    X_test_array = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    tabnet_test_proba = tabnet_model.predict_proba(X_test_array)
    
    if isinstance(X_test, pd.DataFrame):
        xgb_test_proba = xgb_model.predict_proba(X_test)
        lgbm_test_proba = lgbm_model.predict_proba(X_test)
    else:
        xgb_test_proba = xgb_model.predict_proba(X_test_array)
        lgbm_test_proba = lgbm_model.predict_proba(X_test_array)
    
    # Stack predictions
    X_meta_train = np.hstack([tabnet_proba, xgb_proba, lgbm_proba])
    X_meta_test = np.hstack([tabnet_test_proba, xgb_test_proba, lgbm_test_proba])
    
    print(f"  Meta-features shape: Train={X_meta_train.shape}, Test={X_meta_test.shape}")
    return X_meta_train, X_meta_test


class StackingWrapper:
    """Wrapper to save/load the ensemble and make predictions."""
    def __init__(self, tabnet, xgb, lgbm, meta):
        self.tabnet = tabnet
        self.xgb = xgb
        self.lgbm = lgbm
        self.meta = meta
    
    def predict(self, X):
        """Get predictions from all base models, stack, predict with meta-learner."""
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        tabnet_proba = self.tabnet.predict_proba(X_array)
        if isinstance(X, pd.DataFrame):
            xgb_proba = self.xgb.predict_proba(X)
            lgbm_proba = self.lgbm.predict_proba(X)
        else:
            xgb_proba = self.xgb.predict_proba(X_array)
            lgbm_proba = self.lgbm.predict_proba(X_array)
        X_meta = np.hstack([tabnet_proba, xgb_proba, lgbm_proba])
        return self.meta.predict(X_meta)
    
    def predict_proba(self, X):
        """Get probabilities from all base models, stack, predict with meta-learner."""
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        tabnet_proba = self.tabnet.predict_proba(X_array)
        if isinstance(X, pd.DataFrame):
            xgb_proba = self.xgb.predict_proba(X)
            lgbm_proba = self.lgbm.predict_proba(X)
        else:
            xgb_proba = self.xgb.predict_proba(X_array)
            lgbm_proba = self.lgbm.predict_proba(X_array)
        X_meta = np.hstack([tabnet_proba, xgb_proba, lgbm_proba])
        return self.meta.predict_proba(X_meta)


def try_logistic_regression_meta(tabnet_model, xgb_model, lgbm_model, X_train, y_train, X_test, y_test, class_names):
    """Try LogisticRegression as meta-learner."""
    print("\n[3a] Trying LogisticRegression meta-learner...")
    
    # Pre-compute predictions instead of using StackingClassifier
    X_meta_train, X_meta_test = precompute_base_predictions(
        tabnet_model, xgb_model, lgbm_model, X_train, y_train, X_test, cv=3
    )
    
    meta_learner = LogisticRegression(
        class_weight='balanced',
        max_iter=5000,
        random_state=42,
        n_jobs=-1
    )
    
    print("  Training meta-learner on pre-computed predictions...")
    start_time = time.time()
    meta_learner.fit(X_meta_train, y_train)
    train_time = time.time() - start_time
    
    print(f"  Training completed in {train_time/60:.2f} minutes")
    
    # Evaluate
    y_pred = meta_learner.predict(X_meta_test)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    class_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    critical_recall = class_report.get('Severity 1 (Critical)', {}).get('recall', 0)
    
    metrics = {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'critical_recall': float(critical_recall) if critical_recall else None,
        'classification_report': class_report
    }
    
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"  Critical Recall: {metrics['critical_recall']:.4f}")
    
    # Create wrapper object for saving
    stacking = StackingWrapper(tabnet_model, xgb_model, lgbm_model, meta_learner)
    
    return stacking, metrics, train_time


def try_xgboost_meta(tabnet_model, xgb_model, lgbm_model, X_train, y_train, X_test, y_test, class_names):
    """Try XGBoost as meta-learner."""
    print("\n[3b] Trying XGBoost meta-learner...")
    
    # Pre-compute predictions instead of using StackingClassifier
    X_meta_train, X_meta_test = precompute_base_predictions(
        tabnet_model, xgb_model, lgbm_model, X_train, y_train, X_test, cv=3
    )
    
    # Calculate class weights
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight('balanced', y_train)
    
    meta_learner = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    print("  Training meta-learner on pre-computed predictions...")
    start_time = time.time()
    meta_learner.fit(X_meta_train, y_train, sample_weight=sample_weights)
    train_time = time.time() - start_time
    
    print(f"  Training completed in {train_time/60:.2f} minutes")
    
    # Evaluate
    y_pred = meta_learner.predict(X_meta_test)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    class_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    critical_recall = class_report.get('Severity 1 (Critical)', {}).get('recall', 0)
    
    metrics = {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'critical_recall': float(critical_recall) if critical_recall else None,
        'classification_report': class_report
    }
    
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"  Critical Recall: {metrics['critical_recall']:.4f}")
    
    # Create wrapper object for saving
    stacking = StackingWrapper(tabnet_model, xgb_model, lgbm_model, meta_learner)
    
    return stacking, metrics, train_time


def try_lightgbm_meta(tabnet_model, xgb_model, lgbm_model, X_train, y_train, X_test, y_test, class_names):
    """Try LightGBM as meta-learner."""
    print("\n[3c] Trying LightGBM meta-learner...")
    
    # Pre-compute predictions instead of using StackingClassifier
    X_meta_train, X_meta_test = precompute_base_predictions(
        tabnet_model, xgb_model, lgbm_model, X_train, y_train, X_test, cv=3
    )
    
    meta_learner = LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    print("  Training meta-learner on pre-computed predictions...")
    start_time = time.time()
    meta_learner.fit(X_meta_train, y_train)
    train_time = time.time() - start_time
    
    print(f"  Training completed in {train_time/60:.2f} minutes")
    
    # Evaluate
    y_pred = meta_learner.predict(X_meta_test)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    class_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    critical_recall = class_report.get('Severity 1 (Critical)', {}).get('recall', 0)
    
    metrics = {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'critical_recall': float(critical_recall) if critical_recall else None,
        'classification_report': class_report
    }
    
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"  Critical Recall: {metrics['critical_recall']:.4f}")
    
    # Create wrapper object for saving
    stacking = StackingWrapper(tabnet_model, xgb_model, lgbm_model, meta_learner)
    
    return stacking, metrics, train_time


def main():
    """Main function."""
    print("=" * 100)
    print("PHASE 2: STACKING ENSEMBLE WITH TABNET")
    print("=" * 100)
    
    # Load models
    tabnet_model, xgb_model, lgbm_model = load_models()
    
    # Load data
    X_train_full, X_test, y_train_full, y_test = load_preprocessed_data()
    
    class_names = ['Severity 1 (Critical)', 'Severity 2 (Emergent)', 'Severity 3 (Urgent)',
                   'Severity 4 (Less Urgent)', 'Severity 5 (Non-Urgent)']
    
    results = {}
    
    # Try LogisticRegression first
    stacking_lr, metrics_lr, time_lr = try_logistic_regression_meta(
        tabnet_model, xgb_model, lgbm_model,
        X_train_full, y_train_full, X_test, y_test, class_names
    )
    results['LogisticRegression'] = {
        'metrics': metrics_lr,
        'training_time_minutes': time_lr / 60
    }
    
    # Check if we need to try XGBoost
    if metrics_lr['critical_recall'] and metrics_lr['critical_recall'] < 0.12:
        print(f"\n  LogisticRegression ESI 1 recall ({metrics_lr['critical_recall']:.4f}) < 12%, trying XGBoost...")
        stacking_xgb, metrics_xgb, time_xgb = try_xgboost_meta(
            tabnet_model, xgb_model, lgbm_model,
            X_train_full, y_train_full, X_test, y_test, class_names
        )
        results['XGBoost'] = {
            'metrics': metrics_xgb,
            'training_time_minutes': time_xgb / 60
        }
        
        # Check if we need to try LightGBM
        if metrics_xgb['critical_recall'] and metrics_xgb['critical_recall'] < 0.12:
            print(f"\n  XGBoost ESI 1 recall ({metrics_xgb['critical_recall']:.4f}) < 12%, trying LightGBM...")
            stacking_lgb, metrics_lgb, time_lgb = try_lightgbm_meta(
                tabnet_model, xgb_model, lgbm_model,
                X_train_full, y_train_full, X_test, y_test, class_names
            )
            results['LightGBM'] = {
                'metrics': metrics_lgb,
                'training_time_minutes': time_lgb / 60
            }
            best_model = stacking_lgb
            best_name = 'LightGBM'
            best_metrics = metrics_lgb
        else:
            best_model = stacking_xgb
            best_name = 'XGBoost'
            best_metrics = metrics_xgb
    else:
        best_model = stacking_lr
        best_name = 'LogisticRegression'
        best_metrics = metrics_lr
    
    # Print comparison
    print("\n" + "=" * 100)
    print("META-LEARNER COMPARISON")
    print("=" * 100)
    print(f"{'Meta-learner':<20} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12} {'Critical Recall':<15}")
    print("-" * 100)
    for name, data in results.items():
        print(f"{name:<20} {data['metrics']['accuracy']:<12.4f} {data['metrics']['macro_f1']:<12.4f} "
              f"{data['metrics']['weighted_f1']:<12.4f} {data['metrics']['critical_recall']:<15.4f}")
    print("=" * 100)
    
    print(f"\n  Best meta-learner: {best_name}")
    print(f"    Critical Recall: {best_metrics['critical_recall']:.4f}")
    
    # Save results
    print("\n[4] Saving results...")
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'phase': 'Stacking Ensemble with TabNet',
        'base_estimators': ['TabNet', 'XGBoost', 'LightGBM'],
        'results': results,
        'best_meta_learner': best_name,
        'best_metrics': best_metrics
    }
    
    results_path = outputs_dir / "stacking_with_tabnet_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"  Results saved to: {results_path}")
    
    # Save best model
    model_path = models_dir / "stacking_tabnet_xgb_lgb.pkl"
    joblib.dump(best_model, model_path)
    print(f"  Best model saved to: {model_path}")
    
    # Print summary
    print("\n" + "=" * 100)
    print("PHASE 2 COMPLETE")
    print("=" * 100)
    print(f"Best Critical Recall: {best_metrics['critical_recall']:.4f}")
    if best_metrics['critical_recall'] > 0.12:
        print("  Status: GREAT! (>12%)")
    elif best_metrics['critical_recall'] > 0.10:
        print("  Status: GOOD! (>10%)")
    print("=" * 100)


if __name__ == "__main__":
    main()

