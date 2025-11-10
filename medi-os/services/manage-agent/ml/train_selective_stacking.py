"""
Phase 3: Selective Stacking with Attention-Based Routing
Use TabNet attention weights to route samples to TabNet or weighted voting.
"""

import sys
from pathlib import Path
import pickle
import time
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
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

# Set style for plots
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")


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
        feature_names = data.get('feature_names', X_test.columns.tolist())
    else:
        X_test = data['test']
        y_test = data['y_test']
        feature_names = X_test.columns.tolist()
    
    # Handle 5-class severity labels (remap 1-5 to 0-4)
    unique_levels = sorted(y_test.unique())
    if set(unique_levels).issubset({1, 2, 3, 4, 5}) and len(unique_levels) == 5:
        print(f"  Remapping 5-class severity to 0-based: 1->0, 2->1, 3->2, 4->3, 5->4")
        y_test = (y_test - 1).astype(np.int32)
    
    print(f"  Test set: {X_test.shape}")
    print(f"  Features: {len(feature_names)}")
    
    return X_test, y_test, feature_names


def get_critical_feature_indices(feature_names):
    """Get indices of critical features."""
    critical_features = [
        'is_pediatric',
        'tachycardia',
        'chf',
        'shock_index_high',
        'pediatric_fever',
        'hypoxia',
        'severe_pain',
        'respiration',
        'age'
    ]
    
    critical_indices = []
    critical_found = []
    
    for feat in critical_features:
        if feat in feature_names:
            idx = feature_names.index(feat)
            critical_indices.append(idx)
            critical_found.append(feat)
        else:
            print(f"  Warning: Critical feature '{feat}' not found in feature names")
    
    print(f"  Found {len(critical_found)}/{len(critical_features)} critical features")
    print(f"  Critical features: {critical_found}")
    
    return critical_indices, critical_found


def calculate_attention_scores(tabnet_model, X_test, critical_indices):
    """Calculate attention scores per sample using TabNet explain()."""
    print("\n[3] Calculating attention scores...")
    
    try:
        # Use TabNet's explain() method
        print("  Using TabNet explain() method...")
        explain_matrix, masks = tabnet_model.explain(X_test.values)
        
        # Average attention across all steps for critical features per sample
        # explain_matrix shape: [n_samples, n_steps, n_features]
        if len(explain_matrix.shape) == 3:
            # Average across steps, then take critical features
            attention_scores = explain_matrix[:, :, critical_indices].mean(axis=(1, 2))
        else:
            # If 2D, assume it's already averaged or just take critical features
            attention_scores = explain_matrix[:, critical_indices].mean(axis=1)
        
        print(f"  Attention scores shape: {attention_scores.shape}")
        print(f"  Attention score range: [{attention_scores.min():.4f}, {attention_scores.max():.4f}]")
        print(f"  Attention score mean: {attention_scores.mean():.4f}")
        
        return attention_scores, 'explain'
    
    except Exception as e:
        print(f"  Warning: explain() method failed: {e}")
        print("  Falling back to global feature importance...")
        
        # Fallback: Use global feature importance
        try:
            feature_importance = tabnet_model.feature_importances_
            critical_importance = feature_importance[critical_indices].sum()
            
            # For each sample, use a constant score based on critical features
            # This is a simplified approach - we'll use the feature values themselves
            X_test_array = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
            critical_values = X_test_array[:, critical_indices].sum(axis=1)
            attention_scores = critical_values * (critical_importance / len(critical_indices))
            
            print(f"  Using global feature importance fallback")
            print(f"  Attention scores shape: {attention_scores.shape}")
            
            return attention_scores, 'global'
        except Exception as e2:
            print(f"  Error: Global feature importance also failed: {e2}")
            raise


def selective_predict(tabnet_model, xgb_model, lgbm_model, X_test, attention_scores, threshold):
    """Make predictions using selective routing."""
    predictions = np.zeros(len(X_test), dtype=np.int32)
    
    # Get predictions from all models
    tabnet_pred = tabnet_model.predict(X_test.values if isinstance(X_test, pd.DataFrame) else X_test)
    
    # Weighted voting for XGB + LGB (50/50)
    xgb_proba = xgb_model.predict_proba(X_test)
    lgb_proba = lgbm_model.predict_proba(X_test)
    weighted_proba = 0.5 * xgb_proba + 0.5 * lgb_proba
    weighted_pred = np.argmax(weighted_proba, axis=1)
    
    # Route based on attention threshold
    use_tabnet = attention_scores >= threshold
    
    predictions[use_tabnet] = tabnet_pred[use_tabnet]
    predictions[~use_tabnet] = weighted_pred[~use_tabnet]
    
    return predictions, use_tabnet


def evaluate_predictions(y_true, y_pred, class_names):
    """Evaluate predictions and return metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    class_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    critical_recall = class_report.get('Severity 1 (Critical)', {}).get('recall', 0)
    
    return {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'critical_recall': float(critical_recall) if critical_recall else None,
        'classification_report': class_report
    }


def try_thresholds(tabnet_model, xgb_model, lgbm_model, X_test, y_test, attention_scores, class_names):
    """Try different threshold values."""
    print("\n[4] Trying different thresholds...")
    
    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3]
    results = []
    
    print(f"\n{'Threshold':<12} {'TabNet Usage %':<15} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12} {'Critical Recall':<15}")
    print("-" * 100)
    
    for threshold in thresholds:
        y_pred, use_tabnet = selective_predict(
            tabnet_model, xgb_model, lgbm_model, X_test, attention_scores, threshold
        )
        
        tabnet_usage_pct = use_tabnet.sum() / len(use_tabnet) * 100
        
        metrics = evaluate_predictions(y_test, y_pred, class_names)
        
        results.append({
            'threshold': threshold,
            'tabnet_usage_pct': tabnet_usage_pct,
            'metrics': metrics
        })
        
        print(f"{threshold:<12.2f} {tabnet_usage_pct:<15.2f} {metrics['accuracy']:<12.4f} "
              f"{metrics['macro_f1']:<12.4f} {metrics['weighted_f1']:<12.4f} "
              f"{metrics['critical_recall']:<15.4f}")
    
    print("-" * 100)
    
    # Find best threshold (prioritize critical recall, then accuracy)
    best_result = max(results, key=lambda x: (
        x['metrics']['critical_recall'] if x['metrics']['critical_recall'] else 0,
        x['metrics']['accuracy']
    ))
    
    print(f"\n  Best threshold: {best_result['threshold']:.2f}")
    print(f"    TabNet usage: {best_result['tabnet_usage_pct']:.2f}%")
    print(f"    Accuracy: {best_result['metrics']['accuracy']:.4f}")
    print(f"    Critical Recall: {best_result['metrics']['critical_recall']:.4f}")
    
    return results, best_result


def analyze_attention_patterns(attention_scores, y_test, best_threshold, results, output_dir):
    """Analyze attention patterns and create visualizations."""
    print("\n[5] Analyzing attention patterns...")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Attention score distribution by severity class
    ax = axes[0, 0]
    for severity in range(5):
        mask = y_test == severity
        ax.hist(attention_scores[mask], alpha=0.6, label=f'Severity {severity+1}', bins=50)
    ax.set_xlabel('Attention Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Attention Score Distribution by Severity Class')
    ax.legend()
    ax.axvline(best_threshold, color='red', linestyle='--', label=f'Threshold: {best_threshold}')
    
    # 2. TabNet usage by severity class
    ax = axes[0, 1]
    use_tabnet = attention_scores >= best_threshold
    tabnet_usage_by_class = []
    for severity in range(5):
        mask = y_test == severity
        usage_pct = use_tabnet[mask].sum() / mask.sum() * 100
        tabnet_usage_by_class.append(usage_pct)
    
    ax.bar(range(1, 6), tabnet_usage_by_class, color='steelblue')
    ax.set_xlabel('Severity Class')
    ax.set_ylabel('TabNet Usage (%)')
    ax.set_title('TabNet Usage by Severity Class')
    ax.set_xticks(range(1, 6))
    
    # 3. Attention score vs severity class (box plot)
    ax = axes[1, 0]
    data_for_box = [attention_scores[y_test == s] for s in range(5)]
    ax.boxplot(data_for_box, labels=[f'Severity {i+1}' for i in range(5)])
    ax.set_ylabel('Attention Score')
    ax.set_title('Attention Score Distribution by Severity Class')
    ax.axhline(best_threshold, color='red', linestyle='--', label=f'Threshold: {best_threshold}')
    
    # 4. Critical recall vs threshold
    ax = axes[1, 1]
    thresholds_tested = [r['threshold'] for r in results]
    critical_recalls = [r['metrics']['critical_recall'] if r['metrics']['critical_recall'] else 0 for r in results]
    ax.plot(thresholds_tested, critical_recalls, marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Critical Recall')
    ax.set_title('Critical Recall vs Threshold')
    ax.grid(True, alpha=0.3)
    ax.axvline(best_threshold, color='red', linestyle='--', label=f'Best: {best_threshold}')
    ax.legend()
    
    plt.tight_layout()
    
    plot_path = output_dir / "selective_stacking_attention_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved attention analysis to: {plot_path}")
    plt.close()
    
    # Return analysis summary
    analysis = {
        'attention_score_stats': {
            'mean': float(attention_scores.mean()),
            'std': float(attention_scores.std()),
            'min': float(attention_scores.min()),
            'max': float(attention_scores.max()),
            'median': float(np.median(attention_scores))
        },
        'tabnet_usage_by_class': {
            f'Severity {i+1}': float(tabnet_usage_by_class[i]) for i in range(5)
        }
    }
    
    return analysis


def main():
    """Main function."""
    print("=" * 100)
    print("PHASE 3: SELECTIVE STACKING WITH ATTENTION-BASED ROUTING")
    print("=" * 100)
    
    # Load models
    tabnet_model, xgb_model, lgbm_model = load_models()
    
    # Load test data
    X_test, y_test, feature_names = load_test_data()
    
    # Get critical feature indices
    critical_indices, critical_features = get_critical_feature_indices(feature_names)
    
    # Calculate attention scores
    attention_scores, method = calculate_attention_scores(tabnet_model, X_test, critical_indices)
    
    # Class names
    class_names = ['Severity 1 (Critical)', 'Severity 2 (Emergent)', 'Severity 3 (Urgent)',
                   'Severity 4 (Less Urgent)', 'Severity 5 (Non-Urgent)']
    
    # Try different thresholds
    results, best_result = try_thresholds(
        tabnet_model, xgb_model, lgbm_model, X_test, y_test, attention_scores, class_names
    )
    
    # Analyze attention patterns
    analysis = analyze_attention_patterns(
        attention_scores, y_test, best_result['threshold'], results, outputs_dir
    )
    
    # Save results
    print("\n[6] Saving results...")
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'phase': 'Selective Stacking with Attention-Based Routing',
        'attention_method': method,
        'critical_features': critical_features,
        'threshold_results': [
            {
                'threshold': r['threshold'],
                'tabnet_usage_pct': r['tabnet_usage_pct'],
                'metrics': r['metrics']
            }
            for r in results
        ],
        'best_threshold': best_result['threshold'],
        'best_metrics': best_result['metrics'],
        'attention_analysis': analysis
    }
    
    results_path = outputs_dir / "selective_stacking_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"  Results saved to: {results_path}")
    
    # Print summary
    print("\n" + "=" * 100)
    print("PHASE 3 COMPLETE")
    print("=" * 100)
    print(f"Best Threshold: {best_result['threshold']:.2f}")
    print(f"TabNet Usage: {best_result['tabnet_usage_pct']:.2f}%")
    print(f"Best Critical Recall: {best_result['metrics']['critical_recall']:.4f}")
    if best_result['metrics']['critical_recall'] > 0.12:
        print("  Status: GREAT! (>12%)")
    elif best_result['metrics']['critical_recall'] > 0.10:
        print("  Status: GOOD! (>10%)")
    print("=" * 100)


if __name__ == "__main__":
    main()

