"""
Train Final Models: XGBoost, LightGBM, and Stacking Ensemble

Retrains the best models on full training data with optimized hyperparameters,
creates a stacking ensemble, analyzes feature importance, and evaluates on test set.
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
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib

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


def load_best_hyperparameters(results_file):
    """Load best hyperparameters from tuning results."""
    print("\n[1] Loading best hyperparameters...")
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    xgb_params = results['Full_72']['XGBoost']['randomized_search']['best_params']
    lgbm_params = results['Full_72']['LightGBM']['randomized_search']['best_params']
    
    print(f"  XGBoost best params: {xgb_params}")
    print(f"  LightGBM best params: {lgbm_params}")
    
    return xgb_params, lgbm_params


def load_preprocessed_data():
    """Load preprocessed data from cache."""
    print("\n[2] Loading preprocessed data...")
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
    
    # Keep as DataFrames to preserve feature names (needed for LightGBM/XGBoost)
    # Combine train + val for final training
    print(f"\n  Combining train + val for final training...")
    if isinstance(X_train, pd.DataFrame):
        X_train_full = pd.concat([X_train, X_val], ignore_index=True)
        X_test = X_test.copy()  # Keep as DataFrame
    else:
        # If already numpy arrays, convert back to DataFrame with feature names
        X_train_full = pd.DataFrame(
            np.vstack([X_train, X_val]),
            columns=feature_names
        )
        X_test = pd.DataFrame(X_test, columns=feature_names)
    
    if isinstance(y_train, pd.Series):
        y_train_full = pd.concat([y_train, y_val], ignore_index=True)
    else:
        y_train_full = np.concatenate([y_train, y_val])
    print(f"  Final training set: {X_train_full.shape}")
    
    return X_train_full, X_test, y_train_full, y_test, feature_names


def train_xgboost(X_train, y_train, xgb_params):
    """Retrain XGBoost on full data with best hyperparameters."""
    print("\n[3] Training XGBoost on full data...")
    
    # Create model with best params
    xgb = XGBClassifier(**xgb_params)
    
    start_time = time.time()
    xgb.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"  Training completed in {train_time/60:.2f} minutes")
    
    return xgb, train_time


def train_lightgbm(X_train, y_train, lgbm_params):
    """Retrain LightGBM on full data with best hyperparameters."""
    print("\n[4] Training LightGBM on full data...")
    
    # Create model with best params
    lgbm = LGBMClassifier(**lgbm_params)
    
    start_time = time.time()
    lgbm.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"  Training completed in {train_time/60:.2f} minutes")
    
    return lgbm, train_time


def create_stacking_ensemble(xgb_model, lgbm_model):
    """Create stacking ensemble with XGBoost and LightGBM as base estimators."""
    print("\n[5] Creating stacking ensemble...")
    
    # Meta-learner: LogisticRegression with balanced class weights
    meta_learner = LogisticRegression(
        class_weight='balanced',
        max_iter=5000,
        random_state=42,
        n_jobs=-1
    )
    
    # Stacking classifier
    stacking = StackingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('lgbm', lgbm_model)
        ],
        final_estimator=meta_learner,
        cv=3,  # 3-fold CV for faster training (was 5)
        n_jobs=-1,
        verbose=1
    )
    
    print("  Stacking ensemble created:")
    print(f"    Base estimators: XGBoost, LightGBM")
    print(f"    Meta-learner: LogisticRegression (balanced)")
    print(f"    CV folds: 3")
    
    return stacking


def train_stacking_ensemble(stacking, X_train, y_train):
    """Train stacking ensemble on full data."""
    print("\n[6] Training stacking ensemble...")
    
    start_time = time.time()
    stacking.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"  Training completed in {train_time/60:.2f} minutes")
    
    return train_time


def analyze_feature_importance(xgb_model, lgbm_model, feature_names, output_dir):
    """Extract and analyze feature importance from both models."""
    print("\n[7] Analyzing feature importance...")
    
    # Extract feature importances
    xgb_importance = xgb_model.feature_importances_
    lgbm_importance = lgbm_model.feature_importances_
    
    # Create DataFrames
    xgb_df = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_importance,
        'model': 'XGBoost'
    }).sort_values('importance', ascending=False)
    
    lgbm_df = pd.DataFrame({
        'feature': feature_names,
        'importance': lgbm_importance,
        'model': 'LightGBM'
    }).sort_values('importance', ascending=False)
    
    # Combine for comparison
    combined_df = pd.concat([
        xgb_df.assign(model='XGBoost'),
        lgbm_df.assign(model='LightGBM')
    ], ignore_index=True)
    
    # Save to CSV
    importance_csv = output_dir / "final_models_feature_importance.csv"
    combined_df.to_csv(importance_csv, index=False)
    print(f"  Saved feature importance to: {importance_csv}")
    
    # Print top 20 features for each model
    print("\n  Top 20 Features - XGBoost:")
    for idx, row in xgb_df.head(20).iterrows():
        print(f"    {row['feature']:40s}: {row['importance']:.6f}")
    
    print("\n  Top 20 Features - LightGBM:")
    for idx, row in lgbm_df.head(20).iterrows():
        print(f"    {row['feature']:40s}: {row['importance']:.6f}")
    
    # Create visualizations
    create_feature_importance_plots(xgb_df, lgbm_df, output_dir)
    
    return xgb_df, lgbm_df, combined_df


def create_feature_importance_plots(xgb_df, lgbm_df, output_dir):
    """Create feature importance visualization plots."""
    print("\n  Creating feature importance plots...")
    
    # Top 20 features for each model
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # XGBoost plot
    top_xgb = xgb_df.head(20)
    axes[0].barh(range(len(top_xgb)), top_xgb['importance'].values, color='#1f77b4')
    axes[0].set_yticks(range(len(top_xgb)))
    axes[0].set_yticklabels(top_xgb['feature'].values, fontsize=9)
    axes[0].set_xlabel('Importance', fontsize=12)
    axes[0].set_title('Top 20 Features - XGBoost', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)
    
    # LightGBM plot
    top_lgbm = lgbm_df.head(20)
    axes[1].barh(range(len(top_lgbm)), top_lgbm['importance'].values, color='#ff7f0e')
    axes[1].set_yticks(range(len(top_lgbm)))
    axes[1].set_yticklabels(top_lgbm['feature'].values, fontsize=9)
    axes[1].set_xlabel('Importance', fontsize=12)
    axes[1].set_title('Top 20 Features - LightGBM', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / "final_models_feature_importance.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved plot to: {plot_path}")
    plt.close()
    
    # Combined comparison plot (top 15 features)
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Get top 15 features from both models
    top_features_xgb = set(xgb_df.head(15)['feature'].values)
    top_features_lgbm = set(lgbm_df.head(15)['feature'].values)
    all_top_features = list(top_features_xgb.union(top_features_lgbm))
    
    xgb_vals = [xgb_df[xgb_df['feature'] == f]['importance'].values[0] if f in top_features_xgb else 0 
                for f in all_top_features]
    lgbm_vals = [lgbm_df[lgbm_df['feature'] == f]['importance'].values[0] if f in top_features_lgbm else 0 
                 for f in all_top_features]
    
    x = np.arange(len(all_top_features))
    width = 0.35
    
    ax.barh(x - width/2, xgb_vals, width, label='XGBoost', color='#1f77b4', alpha=0.8)
    ax.barh(x + width/2, lgbm_vals, width, label='LightGBM', color='#ff7f0e', alpha=0.8)
    
    ax.set_yticks(x)
    ax.set_yticklabels(all_top_features, fontsize=9)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Feature Importance Comparison: XGBoost vs LightGBM', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    comparison_plot_path = output_dir / "final_models_feature_importance_comparison.png"
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved comparison plot to: {comparison_plot_path}")
    plt.close()


def evaluate_model(model, X_test, y_test, model_name, class_names=None):
    """Evaluate model on test set with comprehensive metrics."""
    print(f"\n[8] Evaluating {model_name} on test set...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
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
    
    # Per-class results
    print(f"\n  Per-class Performance:")
    for i, class_name in enumerate(class_names):
        precision = class_report[class_name]['precision']
        recall = class_report[class_name]['recall']
        f1 = class_report[class_name]['f1-score']
        support = class_report[class_name]['support']
        print(f"    {class_name}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, Support={support}")
    
    return {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'classification_report': class_report,
        'confusion_matrix': cm.tolist(),
        'predictions': y_pred.tolist(),
        'predictions_proba': y_pred_proba.tolist() if y_pred_proba is not None else None
    }


def save_models(xgb_model, lgbm_model, stacking_model, models_dir):
    """Save all trained models."""
    print("\n[9] Saving models...")
    
    models_dir.mkdir(exist_ok=True)
    
    # Save XGBoost
    xgb_path = models_dir / "final_xgboost_full_features.pkl"
    joblib.dump(xgb_model, xgb_path)
    print(f"  Saved XGBoost: {xgb_path}")
    
    # Save LightGBM
    lgbm_path = models_dir / "final_lightgbm_full_features.pkl"
    joblib.dump(lgbm_model, lgbm_path)
    print(f"  Saved LightGBM: {lgbm_path}")
    
    # Save Stacking Ensemble
    stacking_path = models_dir / "final_stacking_ensemble.pkl"
    joblib.dump(stacking_model, stacking_path)
    print(f"  Saved Stacking Ensemble: {stacking_path}")


def create_confusion_matrix_plots(results, output_dir, class_names):
    """Create confusion matrix plots for all models."""
    print("\n  Creating confusion matrix plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    models = ['XGBoost', 'LightGBM', 'Stacking']
    
    for idx, (model_name, ax) in enumerate(zip(models, axes)):
        cm = np.array(results[model_name]['confusion_matrix'])
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={'label': 'Proportion'}
        )
        
        ax.set_title(f'{model_name} - Confusion Matrix', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)
    
    plt.tight_layout()
    cm_path = output_dir / "final_models_confusion_matrices.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"  Saved confusion matrices to: {cm_path}")
    plt.close()


def main():
    print("=" * 100)
    print("FINAL MODELS TRAINING: XGBoost, LightGBM, and Stacking Ensemble")
    print("=" * 100)
    
    # Setup directories
    outputs_dir = project_root / "services" / "manage-agent" / "outputs"
    models_dir = project_root / "services" / "manage-agent" / "models"
    outputs_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    
    # Load best hyperparameters
    results_file = outputs_dir / "hyperparameter_tuning_results.json"
    xgb_params, lgbm_params = load_best_hyperparameters(results_file)
    
    # Load preprocessed data
    X_train_full, X_test, y_train_full, y_test, feature_names = load_preprocessed_data()
    
    # Train XGBoost
    xgb_model, xgb_train_time = train_xgboost(X_train_full, y_train_full, xgb_params)
    
    # Train LightGBM
    lgbm_model, lgbm_train_time = train_lightgbm(X_train_full, y_train_full, lgbm_params)
    
    # Create stacking ensemble
    stacking = create_stacking_ensemble(xgb_model, lgbm_model)
    
    # Train stacking ensemble
    stacking_train_time = train_stacking_ensemble(stacking, X_train_full, y_train_full)
    
    # Analyze feature importance
    xgb_importance_df, lgbm_importance_df, combined_importance_df = analyze_feature_importance(
        xgb_model, lgbm_model, feature_names, outputs_dir
    )
    
    # Evaluate all models on test set
    class_names = ['Severity 1 (Critical)', 'Severity 2 (Emergent)', 'Severity 3 (Urgent)', 
                   'Severity 4 (Less Urgent)', 'Severity 5 (Non-Urgent)']
    
    print("\n" + "=" * 100)
    print("TEST SET EVALUATION")
    print("=" * 100)
    
    xgb_results = evaluate_model(xgb_model, X_test, y_test, "XGBoost", class_names)
    lgbm_results = evaluate_model(lgbm_model, X_test, y_test, "LightGBM", class_names)
    stacking_results = evaluate_model(stacking, X_test, y_test, "Stacking Ensemble", class_names)
    
    # Create confusion matrix plots
    all_results = {
        'XGBoost': xgb_results,
        'LightGBM': lgbm_results,
        'Stacking': stacking_results
    }
    create_confusion_matrix_plots(all_results, outputs_dir, class_names)
    
    # Save models
    save_models(xgb_model, lgbm_model, stacking, models_dir)
    
    # Compile final results
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'models': {
            'XGBoost': {
                'training_time_minutes': xgb_train_time / 60,
                'test_results': xgb_results
            },
            'LightGBM': {
                'training_time_minutes': lgbm_train_time / 60,
                'test_results': lgbm_results
            },
            'Stacking': {
                'training_time_minutes': stacking_train_time / 60,
                'test_results': stacking_results
            }
        },
        'feature_importance': {
            'top_20_xgboost': xgb_importance_df.head(20).to_dict('records'),
            'top_20_lightgbm': lgbm_importance_df.head(20).to_dict('records')
        }
    }
    
    # Save results
    results_path = outputs_dir / "final_models_test_results.json"
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\n[OK] Results saved to: {results_path}")
    
    # Print summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"\n{'Model':<25} {'Test Accuracy':<15} {'Macro F1':<12} {'Weighted F1':<12}")
    print("-" * 70)
    print(f"{'XGBoost':<25} {xgb_results['accuracy']:<15.4f} {xgb_results['macro_f1']:<12.4f} {xgb_results['weighted_f1']:<12.4f}")
    print(f"{'LightGBM':<25} {lgbm_results['accuracy']:<15.4f} {lgbm_results['macro_f1']:<12.4f} {lgbm_results['weighted_f1']:<12.4f}")
    print(f"{'Stacking Ensemble':<25} {stacking_results['accuracy']:<15.4f} {stacking_results['macro_f1']:<12.4f} {stacking_results['weighted_f1']:<12.4f}")
    
    print("\n" + "=" * 100)
    print("FINAL MODELS TRAINING COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()

