"""
Compare and evaluate all trained models.

Generates comprehensive comparison reports, visualizations,
and deployment recommendations.
"""

import sys
from pathlib import Path
import pickle
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'services' / 'manage-agent'))


def map_esi_levels(y):
    """Map ESI levels from [0,1,2,3,4,5,7] to sequential [0,1,2,3,4,5,6]."""
    forward_mapping = {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, 4.0: 4, 5.0: 5, 7.0: 6}
    inverse_mapping = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0, 6: 7.0}
    y_mapped = np.array([forward_mapping[float(val)] for val in y], dtype=np.int32)
    return y_mapped, forward_mapping, inverse_mapping


def load_model(model_name, models_dir):
    """Load a trained model and its metadata."""
    model_path = models_dir / f"{model_name}.pkl"
    
    if not model_path.exists():
        print(f"⚠ Warning: {model_name} not found at {model_path}")
        return None, None
    
    model = joblib.load(model_path)
    
    # Try to load metadata
    metadata_path = models_dir / f"{model_name.replace('_triage', '')}_metadata.pkl"
    if metadata_path.exists():
        metadata = joblib.load(metadata_path)
    else:
        metadata = None
    
    return model, metadata


def evaluate_model(model, X, y, y_original, inverse_mapping, model_name):
    """Evaluate a single model."""
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
    
    acc = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average='macro')
    f1_weighted = f1_score(y, y_pred, average='weighted')
    
    # AUROC
    auc_ovr = None
    if y_pred_proba is not None:
        y_binarized = label_binarize(y, classes=list(range(7)))
        try:
            auc_ovr = roc_auc_score(y_binarized, y_pred_proba, average='macro', multi_class='ovr')
        except:
            auc_ovr = None
    
    # Per-class report
    class_report = classification_report(
        y, y_pred,
        target_names=[f'ESI {inverse_mapping[i]}' for i in range(7)],
        output_dict=True
    )
    
    # ESI 1 recall
    esi_1_recall = class_report.get('ESI 1.0', {}).get('recall', 0.0)
    
    return {
        'model_name': model_name,
        'accuracy': float(acc),
        'macro_f1': float(f1_macro),
        'weighted_f1': float(f1_weighted),
        'auroc': float(auc_ovr) if auc_ovr else None,
        'esi_1_recall': float(esi_1_recall),
        'predictions': y_pred,
        'predictions_proba': y_pred_proba,
        'confusion_matrix': confusion_matrix(y, y_pred)
    }


def plot_confusion_matrices(results, inverse_mapping, output_path):
    """Plot side-by-side confusion matrices for all models."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    esi_labels = [f'ESI {inverse_mapping[i]}' for i in range(7)]
    
    for idx, result in enumerate(results):
        cm = result['confusion_matrix']
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            ax=axes[idx], cbar_kws={'label': 'Count'},
            xticklabels=esi_labels, yticklabels=esi_labels
        )
        axes[idx].set_title(f"{result['model_name']}\n(Acc: {result['accuracy']:.3f})", fontsize=12)
        axes[idx].set_xlabel('Predicted', fontsize=10)
        axes[idx].set_ylabel('Actual', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrices saved: {output_path}")


def plot_model_comparison(results, output_path):
    """Plot bar chart comparing model performance."""
    models = [r['model_name'] for r in results]
    metrics = ['accuracy', 'macro_f1', 'auroc']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(metrics):
        values = [r.get(metric, 0) for r in results]
        if metric == 'auroc' and None in values:
            values = [v if v is not None else 0 for v in values]
        
        axes[idx].bar(models, values, color=['steelblue', 'darkgreen', 'coral'])
        axes[idx].set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Score', fontsize=10)
        axes[idx].set_ylim([0, 1])
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Model comparison chart saved: {output_path}")


def plot_roc_curves(results, y_true, inverse_mapping, output_path):
    """Plot ROC curves for all models (multiclass, one-vs-rest)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    esi_labels = [f'ESI {inverse_mapping[i]}' for i in range(7)]
    y_binarized = label_binarize(y_true, classes=list(range(7)))
    
    for idx, result in enumerate(results):
        if result['predictions_proba'] is None:
            continue
        
        ax = axes[idx] if idx < 4 else None
        if ax is None:
            break
        
        for i in range(7):
            fpr, tpr, _ = roc_curve(y_binarized[:, i], result['predictions_proba'][:, i])
            auc = roc_auc_score(y_binarized[:, i], result['predictions_proba'][:, i])
            ax.plot(fpr, tpr, label=f"{esi_labels[i]} (AUC={auc:.3f})", linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title(f"{result['model_name']} - ROC Curves (OvR)", fontsize=11, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(results), 4):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ ROC curves saved: {output_path}")


def main():
    print("=" * 60)
    print("MODEL COMPARISON & EVALUATION")
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
    
    X_val = data['val']
    X_test = data['test']
    y_val = data['y_val']
    y_test = data['y_test']
    
    # Check if ESI levels are already encoded (v2 cache has them encoded)
    if 7.0 in y_val.unique() or 7 in y_val.unique():
        # Need to map
        y_val_mapped, forward_mapping, inverse_mapping = map_esi_levels(y_val)
        y_test_mapped, _, _ = map_esi_levels(y_test)
    else:
        # Already encoded
        y_val_mapped = y_val.astype(np.int32)
        y_test_mapped = y_test.astype(np.int32)
        forward_mapping = {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, 4.0: 4, 5.0: 5, 7.0: 6}
        inverse_mapping = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0, 6: 7.0}
    
    # Load models
    models_dir = project_root / "services" / "manage-agent" / "models"
    print(f"\nLoading models from: {models_dir}")
    
    models_to_load = [
        ('logistic_regression', 'Logistic Regression'),
        ('random_forest', 'Random Forest'),
        ('xgboost_triage', 'XGBoost')
    ]
    
    loaded_models = {}
    for model_file, model_name in models_to_load:
        model, metadata = load_model(model_file, models_dir)
        if model is not None:
            loaded_models[model_name] = {'model': model, 'metadata': metadata}
            print(f"✓ Loaded: {model_name}")
    
    if not loaded_models:
        raise ValueError("No models found to evaluate!")
    
    # Evaluate all models
    print("\n" + "=" * 60)
    print("EVALUATING MODELS")
    print("=" * 60)
    
    val_results = []
    test_results = []
    
    for model_name, model_data in loaded_models.items():
        model = model_data['model']
        
        print(f"\nEvaluating {model_name}...")
        
        # Validation set
        val_result = evaluate_model(
            model, X_val.values, y_val_mapped, y_val.values,
            inverse_mapping, model_name
        )
        val_results.append(val_result)
        
        # Test set
        test_result = evaluate_model(
            model, X_test.values, y_test_mapped, y_test.values,
            inverse_mapping, model_name
        )
        test_results.append(test_result)
        
        print(f"  Val Accuracy: {val_result['accuracy']:.4f}")
        print(f"  Test Accuracy: {test_result['accuracy']:.4f}")
    
    # Generate comparison table
    print("\n" + "=" * 60)
    print("MODEL COMPARISON TABLE")
    print("=" * 60)
    
    comparison_data = {
        'Model': [r['model_name'] for r in val_results],
        'Val Accuracy': [r['accuracy'] for r in val_results],
        'Val Macro F1': [r['macro_f1'] for r in val_results],
        'Val AUROC': [r['auroc'] if r['auroc'] else 0 for r in val_results],
        'Test Accuracy': [r['accuracy'] for r in test_results],
        'Test Macro F1': [r['macro_f1'] for r in test_results],
        'ESI 1 Recall': [r['esi_1_recall'] for r in val_results]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))
    
    # Save comparison table
    outputs_dir = project_root / "services" / "manage-agent" / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
    # Save JSON
    comparison_json = {
        'validation': {r['model_name']: {
            'accuracy': r['accuracy'],
            'macro_f1': r['macro_f1'],
            'auroc': r['auroc'],
            'esi_1_recall': r['esi_1_recall']
        } for r in val_results},
        'test': {r['model_name']: {
            'accuracy': r['accuracy'],
            'macro_f1': r['macro_f1']
        } for r in test_results}
    }
    
    json_path = outputs_dir / "model_comparison.json"
    with open(json_path, 'w') as f:
        json.dump(comparison_json, f, indent=2)
    print(f"\n✓ Comparison JSON saved: {json_path}")
    
    # Save CSV
    csv_path = outputs_dir / "model_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"✓ Comparison CSV saved: {csv_path}")
    
    # Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    # Confusion matrices
    cm_path = outputs_dir / "confusion_matrix_comparison.png"
    plot_confusion_matrices(val_results, inverse_mapping, cm_path)
    
    # Model comparison chart
    comparison_chart_path = outputs_dir / "model_performance_chart.png"
    plot_model_comparison(val_results, comparison_chart_path)
    
    # ROC curves
    roc_path = outputs_dir / "roc_curves_multiclass.png"
    plot_roc_curves(val_results, y_val_mapped, inverse_mapping, roc_path)
    
    # Deployment recommendation
    print("\n" + "=" * 60)
    print("DEPLOYMENT RECOMMENDATION")
    print("=" * 60)
    
    # Select best model based on validation accuracy + ESI 1 recall
    best_model = max(val_results, key=lambda x: x['accuracy'] + x['esi_1_recall'])
    
    print(f"\nBest Model: {best_model['model_name']}")
    print(f"Validation Accuracy:  {best_model['accuracy']:.4f}")
    print(f"Test Accuracy:        {test_results[val_results.index(best_model)]['accuracy']:.4f}")
    print(f"Macro F1:             {best_model['macro_f1']:.4f}")
    auroc_str = f"{best_model['auroc']:.4f}" if best_model['auroc'] else 'N/A'
    print(f"AUROC:                {auroc_str}")
    print(f"ESI 1 Recall:         {best_model['esi_1_recall']:.4f}")
    
    if best_model['esi_1_recall'] >= 0.85:
        print("\n✓ Model meets ESI 1 recall requirement (≥0.85)")
        print("✓ Model is production-ready for deployment.")
    else:
        print("\n⚠ Warning: ESI 1 recall is below 0.85 threshold")
        print("  Consider additional training or hyperparameter tuning.")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

