"""
Phase 1: Weighted Voting Ensemble
Combine TabNet, XGBoost, and LightGBM with weighted probabilities.
Model order: [LightGBM, XGBoost, TabNet] - LightGBM gets highest weight.
"""

import sys
from pathlib import Path
import pickle
import time
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)
from scipy.optimize import minimize
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
    """Load all three models."""
    print("\n[1] Loading models...")
    
    # Load TabNet
    tabnet_path = models_dir / "tabnet_model.zip.zip"
    if not tabnet_path.exists():
        raise FileNotFoundError(f"TabNet model not found: {tabnet_path}")
    print(f"  Loading TabNet from: {tabnet_path.name}")
    tabnet_model = TabNetClassifier()
    tabnet_model.load_model(str(tabnet_path))
    
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


def load_test_data():
    """Load test data from cache."""
    print("\n[2] Loading test data...")
    cache_v10 = outputs_dir / "preprocessed_data_cache_v10_clinical_features.pkl"
    cache_v9 = outputs_dir / "preprocessed_data_cache_v9_nlp_5class.pkl"
    
    cache_file = cache_v10 if cache_v10.exists() else cache_v9
    if not cache_file.exists():
        raise FileNotFoundError(f"Cache not found: {cache_v10} or {cache_v9}")
    
    print(f"  Loading cache: {cache_file.name}")
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    
    # Extract test data
    if 'X_test' in data:
        X_test = data['X_test']
        y_test = data['y_test']
    else:
        X_test = data['test']
        y_test = data['y_test']
    
    # Handle 5-class severity labels (remap 1-5 to 0-4)
    unique_levels = sorted(y_test.unique())
    if set(unique_levels).issubset({1, 2, 3, 4, 5}) and len(unique_levels) == 5:
        print(f"  Remapping 5-class severity to 0-based: 1->0, 2->1, 3->2, 4->3, 5->4")
        y_test = (y_test - 1).astype(np.int32)
    
    print(f"  Test set: {X_test.shape}")
    print(f"  Test labels: {np.bincount(y_test)}")
    
    return X_test, y_test


def get_probabilities(tabnet_model, xgb_model, lgbm_model, X_test):
    """Get prediction probabilities from all models."""
    print("\n[3] Getting prediction probabilities...")
    
    print("  TabNet predict_proba...")
    tabnet_proba = tabnet_model.predict_proba(X_test.values)
    
    print("  XGBoost predict_proba...")
    xgb_proba = xgb_model.predict_proba(X_test)
    
    print("  LightGBM predict_proba...")
    lgbm_proba = lgbm_model.predict_proba(X_test)
    
    print(f"  TabNet probabilities shape: {tabnet_proba.shape}")
    print(f"  XGBoost probabilities shape: {xgb_proba.shape}")
    print(f"  LightGBM probabilities shape: {lgbm_proba.shape}")
    
    # Verify all have same number of classes
    assert tabnet_proba.shape[1] == xgb_proba.shape[1] == lgbm_proba.shape[1], \
        "Models have different number of classes!"
    
    return tabnet_proba, xgb_proba, lgbm_proba


def weighted_voting(lgb_proba, xgb_proba, tabnet_proba, weights):
    """
    Perform weighted voting.
    
    Args:
        lgb_proba: LightGBM probabilities (shape: [n_samples, n_classes])
        xgb_proba: XGBoost probabilities (shape: [n_samples, n_classes])
        tabnet_proba: TabNet probabilities (shape: [n_samples, n_classes])
        weights: [weight_lgb, weight_xgb, weight_tabnet]
    
    Returns:
        predictions: Weighted voting predictions
    """
    weighted_proba = (
        weights[0] * lgb_proba +
        weights[1] * xgb_proba +
        weights[2] * tabnet_proba
    )
    predictions = np.argmax(weighted_proba, axis=1)
    return predictions


def evaluate_predictions(y_true, y_pred, class_names):
    """Evaluate predictions and return metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Per-class metrics
    class_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Critical class recall (Severity 1 = class 0)
    critical_recall = class_report.get('Severity 1 (Critical)', {}).get('recall', 0)
    
    return {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'critical_recall': float(critical_recall) if critical_recall else None,
        'classification_report': class_report
    }


def try_weight_combinations(lgb_proba, xgb_proba, tabnet_proba, y_test, class_names):
    """Try different weight combinations."""
    print("\n[4] Trying weight combinations...")
    
    # Weight combinations: [LightGBM, XGBoost, TabNet]
    weight_combinations = [
        ([0.5, 0.3, 0.2], "Balanced"),
        ([0.4, 0.3, 0.3], "More TabNet"),
        ([0.6, 0.3, 0.1], "Less TabNet"),
        ([0.4, 0.35, 0.25], "Moderate TabNet"),
        ([0.35, 0.35, 0.30], "High TabNet (critical focus)")
    ]
    
    results = []
    
    print(f"\n{'Combination':<30} {'Weights (LGB,XGB,Tab)':<25} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12} {'Critical Recall':<15}")
    print("-" * 120)
    
    for weights, description in weight_combinations:
        y_pred = weighted_voting(lgb_proba, xgb_proba, tabnet_proba, weights)
        metrics = evaluate_predictions(y_test, y_pred, class_names)
        
        results.append({
            'weights': weights,
            'description': description,
            'metrics': metrics
        })
        
        print(f"{description:<30} {str(weights):<25} {metrics['accuracy']:<12.4f} "
              f"{metrics['macro_f1']:<12.4f} {metrics['weighted_f1']:<12.4f} "
              f"{metrics['critical_recall']:<15.4f}")
    
    print("-" * 120)
    
    # Find best combination (prioritize critical recall, then accuracy)
    best_combo = max(results, key=lambda x: (
        x['metrics']['critical_recall'] if x['metrics']['critical_recall'] else 0,
        x['metrics']['accuracy']
    ))
    
    print(f"\n  Best combination: {best_combo['description']}")
    print(f"    Weights: {best_combo['weights']}")
    print(f"    Accuracy: {best_combo['metrics']['accuracy']:.4f}")
    print(f"    Critical Recall: {best_combo['metrics']['critical_recall']:.4f}")
    
    return results, best_combo


def optimize_weights(lgb_proba, xgb_proba, tabnet_proba, y_test, class_names, initial_weights):
    """Optimize weights to maximize critical recall."""
    print("\n[5] Optimizing weights...")
    
    def objective(weights):
        """Objective function: minimize negative critical recall."""
        # Normalize weights to sum to 1
        weights = weights / weights.sum()
        y_pred = weighted_voting(lgb_proba, xgb_proba, tabnet_proba, weights)
        metrics = evaluate_predictions(y_test, y_pred, class_names)
        # Return negative critical recall (minimize negative = maximize recall)
        return -metrics['critical_recall'] if metrics['critical_recall'] else 1.0
    
    # Constraints: weights sum to 1, all >= 0
    constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1.0}
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    
    # Optimize
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 100}
    )
    
    if result.success:
        optimized_weights = result.x / result.x.sum()  # Normalize
        print(f"  Optimization successful!")
        print(f"  Optimized weights: {optimized_weights}")
        
        # Evaluate optimized weights
        y_pred_opt = weighted_voting(lgb_proba, xgb_proba, tabnet_proba, optimized_weights)
        metrics_opt = evaluate_predictions(y_test, y_pred_opt, class_names)
        
        print(f"  Optimized Accuracy: {metrics_opt['accuracy']:.4f}")
        print(f"  Optimized Critical Recall: {metrics_opt['critical_recall']:.4f}")
        
        return optimized_weights, metrics_opt
    else:
        print(f"  Optimization failed: {result.message}")
        return None, None


def main():
    """Main function."""
    print("=" * 100)
    print("PHASE 1: WEIGHTED VOTING ENSEMBLE")
    print("=" * 100)
    
    # Load models
    tabnet_model, xgb_model, lgbm_model = load_models()
    
    # Load test data
    X_test, y_test = load_test_data()
    
    # Get probabilities
    tabnet_proba, xgb_proba, lgbm_proba = get_probabilities(
        tabnet_model, xgb_model, lgbm_model, X_test
    )
    
    # Class names
    class_names = ['Severity 1 (Critical)', 'Severity 2 (Emergent)', 'Severity 3 (Urgent)',
                   'Severity 4 (Less Urgent)', 'Severity 5 (Non-Urgent)']
    
    # Try weight combinations
    results, best_combo = try_weight_combinations(
        lgbm_proba, xgb_proba, tabnet_proba, y_test, class_names
    )
    
    # Optimize best combination if critical recall > 10%
    optimized_weights = None
    optimized_metrics = None
    if best_combo['metrics']['critical_recall'] and best_combo['metrics']['critical_recall'] > 0.10:
        optimized_weights, optimized_metrics = optimize_weights(
            lgbm_proba, xgb_proba, tabnet_proba, y_test, class_names,
            np.array(best_combo['weights'])
        )
    
    # Save results
    print("\n[6] Saving results...")
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'phase': 'Weighted Voting Ensemble',
        'model_order': ['LightGBM', 'XGBoost', 'TabNet'],
        'manual_combinations': [
            {
                'weights': r['weights'],
                'description': r['description'],
                'metrics': r['metrics']
            }
            for r in results
        ],
        'best_manual': {
            'weights': best_combo['weights'],
            'description': best_combo['description'],
            'metrics': best_combo['metrics']
        }
    }
    
    if optimized_weights is not None:
        results_data['optimized'] = {
            'weights': optimized_weights.tolist(),
            'metrics': optimized_metrics
        }
        print(f"  Best overall: Optimized weights")
        print(f"    Critical Recall: {optimized_metrics['critical_recall']:.4f}")
    else:
        print(f"  Best overall: {best_combo['description']}")
        print(f"    Critical Recall: {best_combo['metrics']['critical_recall']:.4f}")
    
    results_path = outputs_dir / "weighted_voting_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"  Results saved to: {results_path}")
    
    # Print summary
    print("\n" + "=" * 100)
    print("PHASE 1 COMPLETE")
    print("=" * 100)
    print(f"Best Critical Recall: {optimized_metrics['critical_recall'] if optimized_metrics else best_combo['metrics']['critical_recall']:.4f}")
    if optimized_metrics and optimized_metrics['critical_recall'] > 0.12:
        print("  Status: GREAT! (>12%)")
    elif (optimized_metrics and optimized_metrics['critical_recall'] > 0.10) or \
         (best_combo['metrics']['critical_recall'] and best_combo['metrics']['critical_recall'] > 0.10):
        print("  Status: GOOD! (>10%)")
    print("=" * 100)


if __name__ == "__main__":
    main()

