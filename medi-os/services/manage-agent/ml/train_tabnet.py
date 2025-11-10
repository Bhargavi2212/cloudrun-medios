"""
Train TabNet Model with Attention Mechanism

TabNet is a deep learning model designed for tabular data with:
- Sequential attention mechanism
- Automatic feature selection
- Interpretable attention weights
- Better handling of complex feature interactions
"""

import sys
from pathlib import Path
import pickle
import time
import json
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'services' / 'manage-agent'))

# Set style for plots
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")


def load_preprocessed_data():
    """Load preprocessed data from cache."""
    print("\n[1] Loading preprocessed data...")
    # Try v10 first (with clinical features), fallback to v9
    cache_v10 = project_root / "services" / "manage-agent" / "outputs" / "preprocessed_data_cache_v10_clinical_features.pkl"
    cache_v9 = project_root / "services" / "manage-agent" / "outputs" / "preprocessed_data_cache_v9_nlp_5class.pkl"
    
    cache_file = cache_v10 if cache_v10.exists() else cache_v9
    if not cache_file.exists():
        raise FileNotFoundError(f"Cache not found: {cache_v10} or {cache_v9}")
    
    print(f"  Loading cache: {cache_file.name}")
    
    with open(cache_file, 'rb') as f:
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
    
    print(f"  Training set: {X_train.shape}")
    print(f"  Validation set: {X_val.shape}")
    print(f"  Test set: {X_test.shape}")
    print(f"  Features: {len(feature_names)}")
    
    # Handle 5-class severity labels (remap 1-5 to 0-4)
    unique_levels = sorted(y_train.unique())
    if set(unique_levels).issubset({1, 2, 3, 4, 5}) and len(unique_levels) == 5:
        print(f"\n  Remapping 5-class severity to 0-based: 1->0, 2->1, 3->2, 4->3, 5->4")
        y_train = (y_train - 1).astype(np.int32)
        y_val = (y_val - 1).astype(np.int32)
        y_test = (y_test - 1).astype(np.int32)
    
    # Convert to numpy arrays (TabNet works with numpy)
    print(f"\n  Combining train + val for final training...")
    if isinstance(X_train, pd.DataFrame):
        X_train_full = pd.concat([X_train, X_val], ignore_index=True)
        X_train_full = X_train_full.values
        X_test = X_test.values
    else:
        X_train_full = np.vstack([X_train, X_val])
        X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    
    if isinstance(y_train, pd.Series):
        y_train_full = pd.concat([y_train, y_val], ignore_index=True).values
        y_test = y_test.values if isinstance(y_test, pd.Series) else y_test
    else:
        y_train_full = np.concatenate([y_train, y_val])
    
    print(f"  Final training set: {X_train_full.shape}")
    print(f"  Test set: {X_test.shape}")
    
    return X_train_full, X_test, y_train_full, y_test, feature_names


def train_tabnet(X_train, y_train, X_val, y_val):
    """
    Train TabNet classifier with hyperparameter tuning.
    
    Args:
        X_train: Training features (numpy array)
        y_train: Training labels (numpy array)
        X_val: Validation features (numpy array)
        y_val: Validation labels (numpy array)
    
    Returns:
        Trained TabNet model and training time
    """
    print("\n[2] Training TabNet...")
    
    # TabNet hyperparameters
    # Start with reasonable defaults, can tune later
    tabnet_params = {
        'n_d': 64,              # Dimension of feature representation
        'n_a': 64,              # Dimension of attention embedding
        'n_steps': 5,           # Number of steps in encoder
        'gamma': 1.5,           # Coefficient for feature reusage
        'lambda_sparse': 1e-3,  # Sparsity regularization
        'optimizer_fn': torch.optim.Adam,  # Must be callable, not string
        'optimizer_params': {'lr': 2e-2},
        'mask_type': 'entmax',  # 'sparsemax' or 'entmax'
        'n_shared': 2,          # Number of shared steps
        'n_independent': 2,     # Number of independent steps
        'epsilon': 1e-15,
        'seed': 42,
        'verbose': 1,
        'device_name': 'cpu'     # Use CPU (change to 'cuda' if GPU available)
    }
    
    # Calculate class weights for imbalance
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights_dict = dict(zip(classes, class_weights))
    
    print(f"  Class weights: {class_weights_dict}")
    print(f"  TabNet parameters:")
    print(f"    n_d (feature dimension): {tabnet_params['n_d']}")
    print(f"    n_a (attention dimension): {tabnet_params['n_a']}")
    print(f"    n_steps: {tabnet_params['n_steps']}")
    print(f"    gamma (feature reusage): {tabnet_params['gamma']}")
    
    # Create TabNet model
    tabnet = TabNetClassifier(**tabnet_params)
    
    # Train with early stopping
    start_time = time.time()
    
    tabnet.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_name=['val'],
        eval_metric=['accuracy'],
        max_epochs=200,
        patience=10,  # Early stopping patience (balanced)
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        weights=class_weights_dict  # Class weights for imbalance
    )
    
    train_time = time.time() - start_time
    print(f"\n  Training completed in {train_time/60:.2f} minutes")
    
    return tabnet, train_time


def evaluate_model(model, X_test, y_test, feature_names, class_names=None):
    """Evaluate TabNet model on test set."""
    print("\n[3] Evaluating TabNet on test set...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Per-class metrics
    class_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    
    # Critical class recall (Severity 1 = class 0)
    critical_recall = class_report.get('Severity 1 (Critical)', {}).get('recall', 0)
    if critical_recall:
        print(f"  Critical (Severity 1) Recall: {critical_recall:.4f}")
    
    # Print per-class performance
    print(f"\n  Per-class Performance:")
    for class_name in class_names:
        if class_name in class_report:
            metrics = class_report[class_name]
            print(f"    {class_name}:")
            print(f"      Precision: {metrics['precision']:.4f}")
            print(f"      Recall: {metrics['recall']:.4f}")
            print(f"      F1-Score: {metrics['f1-score']:.4f}")
    
    results = {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'critical_recall': float(critical_recall) if critical_recall else None,
        'classification_report': class_report,
        'confusion_matrix': cm.tolist()
    }
    
    return results


def visualize_attention(model, X_sample, feature_names, output_dir):
    """
    Visualize TabNet attention weights.
    
    Args:
        model: Trained TabNet model
        X_sample: Sample data for attention visualization
        feature_names: List of feature names
        output_dir: Directory to save visualizations
    """
    print("\n[4] Visualizing TabNet attention weights...")
    
    # Get attention masks for sample
    # TabNet provides explain_matrix method
    try:
        explain_matrix, masks = model.explain(X_sample[:1000])  # Use first 1000 samples
        
        # Average attention across samples
        avg_attention = explain_matrix.mean(axis=0)
        
        # Create DataFrame
        attention_df = pd.DataFrame({
            'feature': feature_names,
            'attention': avg_attention
        }).sort_values('attention', ascending=False)
        
        # Save attention weights
        attention_csv = output_dir / "tabnet_attention_weights.csv"
        attention_df.to_csv(attention_csv, index=False)
        print(f"  Saved attention weights to: {attention_csv}")
        
        # Plot top 20 features
        plt.figure(figsize=(12, 8))
        top_20 = attention_df.head(20)
        plt.barh(range(len(top_20)), top_20['attention'].values, color='steelblue')
        plt.yticks(range(len(top_20)), top_20['feature'].values)
        plt.xlabel('Average Attention Weight', fontsize=12)
        plt.title('TabNet: Top 20 Features by Attention Weight', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        plot_path = output_dir / "tabnet_attention_weights.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Saved attention plot to: {plot_path}")
        plt.close()
        
        return attention_df
        
    except Exception as e:
        print(f"  Warning: Could not generate attention visualization: {e}")
        return None


def create_confusion_matrix_plot(results, output_dir, class_names):
    """Create confusion matrix visualization."""
    print("\n[5] Creating confusion matrix plot...")
    
    cm = np.array(results['confusion_matrix'])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, values_format='d', cmap='Blues')
    ax.set_title('TabNet: Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = output_dir / "tabnet_confusion_matrix.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved confusion matrix to: {plot_path}")
    plt.close()


def compare_with_baseline(tabnet_results, baseline_file, output_dir):
    """Compare TabNet results with baseline models."""
    print("\n[6] Comparing with baseline models...")
    
    try:
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
        
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        print(f"{'Model':<20} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12} {'Critical Recall':<15}")
        print("-" * 80)
        
        # Baseline models
        for model_name in ['XGBoost', 'LightGBM', 'Stacking']:
            if model_name in baseline_data.get('models', {}):
                results = baseline_data['models'][model_name]['test_results']
                crit_recall = results.get('critical_recall', 
                    results.get('classification_report', {}).get('Severity 1 (Critical)', {}).get('recall', 0))
                print(f"{model_name:<20} {results['accuracy']:<12.4f} {results['macro_f1']:<12.4f} "
                      f"{results['weighted_f1']:<12.4f} {crit_recall:<15.4f}")
        
        # TabNet
        crit_recall = tabnet_results.get('critical_recall', 0)
        print(f"{'TabNet':<20} {tabnet_results['accuracy']:<12.4f} {tabnet_results['macro_f1']:<12.4f} "
              f"{tabnet_results['weighted_f1']:<12.4f} {crit_recall:<15.4f}")
        
        print("=" * 80)
        
    except Exception as e:
        print(f"  Warning: Could not compare with baseline: {e}")


def main():
    """Main training function."""
    print("=" * 100)
    print("TABNET TRAINING: Deep Learning with Attention Mechanism")
    print("=" * 100)
    
    # Setup directories
    outputs_dir = project_root / "services" / "manage-agent" / "outputs"
    models_dir = project_root / "services" / "manage-agent" / "models"
    outputs_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    
    # Load preprocessed data
    X_train_full, X_test, y_train_full, y_test, feature_names = load_preprocessed_data()
    
    # Split validation set from training (use 10% for validation)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.1,
        random_state=42,
        stratify=y_train_full
    )
    
    print(f"\n  Training set: {X_train.shape}")
    print(f"  Validation set: {X_val.shape}")
    print(f"  Test set: {X_test.shape}")
    
    # Train TabNet
    tabnet_model, train_time = train_tabnet(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    class_names = ['Severity 1 (Critical)', 'Severity 2 (Emergent)', 'Severity 3 (Urgent)', 
                   'Severity 4 (Less Urgent)', 'Severity 5 (Non-Urgent)']
    
    test_results = evaluate_model(tabnet_model, X_test, y_test, feature_names, class_names)
    
    # Visualize attention
    attention_df = visualize_attention(tabnet_model, X_test, feature_names, outputs_dir)
    
    # Create confusion matrix
    create_confusion_matrix_plot(test_results, outputs_dir, class_names)
    
    # Compare with baseline
    baseline_file = outputs_dir / "final_models_test_results.json"
    compare_with_baseline(test_results, baseline_file, outputs_dir)
    
    # Save model
    model_path = models_dir / "tabnet_model.zip"
    tabnet_model.save_model(str(model_path))
    print(f"\n[7] Model saved to: {model_path}")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'model': 'TabNet',
        'training_time_minutes': train_time / 60,
        'test_results': test_results,
        'feature_count': len(feature_names),
        'attention_weights': attention_df.to_dict('records') if attention_df is not None else None
    }
    
    results_path = outputs_dir / "tabnet_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {results_path}")
    
    # Print summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"TabNet Performance:")
    print(f"  Accuracy: {test_results['accuracy']:.4f}")
    print(f"  Macro F1: {test_results['macro_f1']:.4f}")
    print(f"  Weighted F1: {test_results['weighted_f1']:.4f}")
    if test_results.get('critical_recall'):
        print(f"  Critical Recall: {test_results['critical_recall']:.4f}")
    print(f"  Training Time: {train_time/60:.2f} minutes")
    
    print("\n" + "=" * 100)
    print("TABNET TRAINING COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()

