"""
Hyperparameter Tuning for All Models

Optimizes hyperparameters using:
- RandomizedSearchCV for all models (5k samples max, 3-fold CV)

All models are tuned on both Full Features (72) and Age+RFV+Vitals (52) feature sets.
"""

import sys
from pathlib import Path
import pickle
import time
import json
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    make_scorer
)
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'services' / 'manage-agent'))


def filter_age_rfv_vitals_features(X_train, X_val, X_test, feature_names):
    """Filter to Age + RFV + Vitals features."""
    age_features = ['age']
    rfv1_emb_features = [f for f in feature_names if f.startswith('rfv1_emb_')]
    rfv2_emb_features = [f for f in feature_names if f.startswith('rfv2_emb_')]
    rfv_3d_features = [f for f in feature_names if f in ['rfv1_3d', 'rfv2_3d']]
    vital_features = ['pulse', 'respiration', 'sbp', 'dbp', 'o2_sat', 'temp_c', 'pain', 'gcs', 'on_oxygen']
    vital_features = [f for f in vital_features if f in feature_names]
    
    selected_features = age_features + rfv1_emb_features + rfv2_emb_features + rfv_3d_features + vital_features
    
    if isinstance(X_train, pd.DataFrame):
        X_train_filtered = X_train[selected_features]
        X_val_filtered = X_val[selected_features]
        X_test_filtered = X_test[selected_features]
    else:
        feature_indices = [feature_names.index(f) for f in selected_features]
        X_train_filtered = X_train[:, feature_indices]
        X_val_filtered = X_val[:, feature_indices]
        X_test_filtered = X_test[:, feature_indices]
    
    return X_train_filtered, X_val_filtered, X_test_filtered, selected_features


def get_parameter_grids():
    """Define parameter grids for all models."""
    param_grids = {
        'LogisticRegression': {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'solver': ['lbfgs', 'saga'],
            'penalty': ['l2'],
            'max_iter': [2000, 3000, 5000, 10000],
            'tol': [1e-4, 1e-3, 1e-2],
            'class_weight': ['balanced'],
            'random_state': [42],
            'n_jobs': [-1]
        },
        'RandomForest': {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 15, 20, 25, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'class_weight': ['balanced'],
            'random_state': [42],
            'n_jobs': [-1]
        },
        'XGBoost': {
            'n_estimators': [200, 300, 400, 500],
            'max_depth': [6, 8, 10, 12],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': [0, 0.5, 1.0],
            'reg_lambda': [0, 1.0, 2.0],
            'eval_metric': ['mlogloss'],
            'tree_method': ['hist'],
            'random_state': [42],
            'n_jobs': [-1]
        },
        'LightGBM': {
            'n_estimators': [200, 300, 400, 500],
            'max_depth': [6, 8, 10, 12],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': [0, 0.5, 1.0],
            'reg_lambda': [0, 1.0, 2.0],
            'objective': ['multiclass'],
            'metric': ['multi_logloss'],
            'random_state': [42],
            'n_jobs': [-1],
            'verbose': [-1]
        }
    }
    return param_grids


def create_model(model_name, params=None):
    """Create model instance."""
    if params is None:
        params = {}
    
    if model_name == 'LogisticRegression':
        # Ensure max_iter is high enough and tol is set for convergence
        if 'max_iter' not in params:
            params['max_iter'] = 5000
        if 'tol' not in params:
            params['tol'] = 1e-3
        return LogisticRegression(**params)
    elif model_name == 'RandomForest':
        return RandomForestClassifier(**params)
    elif model_name == 'XGBoost':
        return XGBClassifier(**params)
    elif model_name == 'LightGBM':
        return LGBMClassifier(**params)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_randomized_search(model_name, X_train, y_train, param_grid, n_iter=50, cv=3):
    """Run RandomizedSearchCV."""
    print(f"\n[RandomizedSearchCV] {model_name}")
    print(f"  Parameters: {len(param_grid)} parameter combinations")
    print(f"  Iterations: {n_iter}")
    print(f"  CV folds: {cv}")
    
    # Sample data for all models to speed up training (use max 5k samples)
    X_train_actual = X_train
    y_train_actual = y_train
    original_size = len(X_train)
    sample_size = 5000  # Use max 5k samples for all models
    
    if original_size > sample_size:
        print(f"  [SAMPLING] Using {sample_size:,} samples (from {original_size:,}) for faster training")
        try:
            # Use stratified sampling to maintain class distribution
            X_train_actual, _, y_train_actual, _ = train_test_split(
                X_train, y_train,
                train_size=sample_size,
                stratify=y_train,
                random_state=42
            )
            print(f"  [SAMPLING] Sampled {len(X_train_actual):,} samples with stratified sampling")
        except ValueError as e:
            # If stratified sampling fails (e.g., some classes too small), use random sampling
            print(f"  [SAMPLING] Stratified sampling failed, using random sampling: {e}")
            np.random.seed(42)
            indices = np.random.choice(original_size, size=min(sample_size, original_size), replace=False)
            X_train_actual = X_train[indices]
            y_train_actual = y_train[indices]
            print(f"  [SAMPLING] Randomly sampled {len(X_train_actual):,} samples")
    
    model = create_model(model_name)
    
    # Create scorer for both accuracy and macro F1
    scorers = {
        'accuracy': 'accuracy',
        'f1_macro': make_scorer(f1_score, average='macro')
    }
    
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=2,
        return_train_score=True
    )
    
    start_time = time.time()
    random_search.fit(X_train_actual, y_train_actual)
    search_time = time.time() - start_time
    
    print(f"  Best score: {random_search.best_score_:.4f}")
    print(f"  Best params: {random_search.best_params_}")
    print(f"  Search time: {search_time/60:.1f} minutes")
    
    return random_search, search_time


def run_grid_search(model_name, X_train, y_train, param_grid, cv=3):
    """Run GridSearchCV on full parameter grid."""
    print(f"\n[GridSearchCV] {model_name}")
    print(f"  Parameters: {len(param_grid)} parameter combinations")
    print(f"  CV folds: {cv}")
    
    # Sample data for all models to speed up training (use max 5k samples)
    X_train_actual = X_train
    y_train_actual = y_train
    original_size = len(X_train)
    sample_size = 5000  # Use max 5k samples for all models
    
    if original_size > sample_size:
        print(f"  [SAMPLING] Using {sample_size:,} samples (from {original_size:,}) for faster training")
        try:
            # Use stratified sampling to maintain class distribution
            X_train_actual, _, y_train_actual, _ = train_test_split(
                X_train, y_train,
                train_size=sample_size,
                stratify=y_train,
                random_state=42
            )
            print(f"  [SAMPLING] Sampled {len(X_train_actual):,} samples with stratified sampling")
        except ValueError as e:
            # If stratified sampling fails (e.g., some classes too small), use random sampling
            print(f"  [SAMPLING] Stratified sampling failed, using random sampling: {e}")
            np.random.seed(42)
            indices = np.random.choice(original_size, size=min(sample_size, original_size), replace=False)
            X_train_actual = X_train[indices]
            y_train_actual = y_train[indices]
            print(f"  [SAMPLING] Randomly sampled {len(X_train_actual):,} samples")
    
    # Calculate total combinations
    total_combinations = 1
    for param, values in param_grid.items():
        if isinstance(values, list):
            total_combinations *= len(values)
        else:
            total_combinations *= 1
    
    print(f"  Total combinations: {total_combinations:,}")
    
    model = create_model(model_name)
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )
    
    start_time = time.time()
    grid_search.fit(X_train_actual, y_train_actual)
    search_time = time.time() - start_time
    
    print(f"  Best score: {grid_search.best_score_:.4f}")
    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Search time: {search_time/60:.1f} minutes")
    
    return grid_search, search_time


def run_grid_search_refinement(model_name, X_train, y_train, best_params, param_grid, cv=3):
    """Run GridSearchCV refinement around best parameters."""
    print(f"\n[GridSearchCV Refinement] {model_name}")
    
    # Skip Random Forest - RandomizedSearchCV results are sufficient
    if model_name == 'RandomForest':
        print(f"  [SKIP] Skipping GridSearchCV for Random Forest (memory-intensive)")
        print(f"  [INFO] RandomizedSearchCV results are typically sufficient for Random Forest")
        return None, 0.0
    
    # Sample data for all models to speed up training (use max 5k samples)
    X_train_actual = X_train
    y_train_actual = y_train
    original_size = len(X_train)
    sample_size = 5000  # Use max 5k samples for all models
    
    if original_size > sample_size:
        print(f"  [SAMPLING] Using {sample_size:,} samples (from {original_size:,}) for faster training")
        try:
            # Use stratified sampling to maintain class distribution
            X_train_actual, _, y_train_actual, _ = train_test_split(
                X_train, y_train,
                train_size=sample_size,
                stratify=y_train,
                random_state=42
            )
            print(f"  [SAMPLING] Sampled {len(X_train_actual):,} samples with stratified sampling")
        except ValueError as e:
            # If stratified sampling fails (e.g., some classes too small), use random sampling
            print(f"  [SAMPLING] Stratified sampling failed, using random sampling: {e}")
            np.random.seed(42)
            indices = np.random.choice(original_size, size=min(sample_size, original_size), replace=False)
            X_train_actual = X_train[indices]
            y_train_actual = y_train[indices]
            print(f"  [SAMPLING] Randomly sampled {len(X_train_actual):,} samples")
    
    # Create refined grid around best parameters
    refined_grid = {}
    for param, best_value in best_params.items():
        if param in param_grid:
            if isinstance(best_value, (int, float)):
                # Create smaller range around best value
                if param == 'C':
                    refined_grid[param] = [best_value * 0.5, best_value, best_value * 2]
                elif param == 'learning_rate':
                    refined_grid[param] = [max(0.001, best_value * 0.5), best_value, min(0.3, best_value * 2)]
                elif param == 'max_iter':
                    # For Logistic Regression, ensure high enough max_iter
                    if best_value < 5000:
                        refined_grid[param] = [max(2000, best_value), best_value + 2000, best_value + 5000]
                    else:
                        refined_grid[param] = [best_value, best_value + 2000, best_value + 5000]
                elif param == 'tol':
                    # For convergence tolerance, keep best value and nearby
                    if best_value <= 1e-4:
                        refined_grid[param] = [1e-4, 1e-3, 1e-2]
                    elif best_value <= 1e-3:
                        refined_grid[param] = [1e-4, 1e-3, 1e-2]
                    else:
                        refined_grid[param] = [1e-3, 1e-2, 1e-1]
                elif param in ['n_estimators', 'max_depth']:
                    # Use smaller range
                    if isinstance(best_value, int):
                        refined_grid[param] = [max(1, best_value - 50), best_value, best_value + 50]
                    else:
                        refined_grid[param] = param_grid[param][:3]  # Take first 3
                else:
                    refined_grid[param] = [best_value]
            else:
                refined_grid[param] = [best_value]
        else:
            refined_grid[param] = [best_value]
    
    # Add a few more values for key parameters
    if 'n_estimators' in refined_grid and len(refined_grid['n_estimators']) == 1:
        refined_grid['n_estimators'] = [refined_grid['n_estimators'][0] - 50, 
                                        refined_grid['n_estimators'][0],
                                        refined_grid['n_estimators'][0] + 50]
    
    print(f"  Refined grid size: {sum(len(v) if isinstance(v, list) else 1 for v in refined_grid.values())} combinations")
    
    model = create_model(model_name)
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=refined_grid,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )
    
    start_time = time.time()
    grid_search.fit(X_train_actual, y_train_actual)
    search_time = time.time() - start_time
    
    print(f"  Best score: {grid_search.best_score_:.4f}")
    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Search time: {search_time/60:.1f} minutes")
    
    return grid_search, search_time


def final_validation(model, model_name, X_train, y_train, cv=3):
    """Final 3-fold validation on best model."""
    print(f"\n[Final Validation] {model_name} (3-fold CV)")
    
    # Sample data for faster validation (use max 5k samples)
    X_train_actual = X_train
    y_train_actual = y_train
    original_size = len(X_train)
    sample_size = 5000  # Use max 5k samples
    
    if original_size > sample_size:
        print(f"  [SAMPLING] Using {sample_size:,} samples (from {original_size:,}) for faster validation")
        try:
            # Use stratified sampling to maintain class distribution
            X_train_actual, _, y_train_actual, _ = train_test_split(
                X_train, y_train,
                train_size=sample_size,
                stratify=y_train,
                random_state=42
            )
            print(f"  [SAMPLING] Sampled {len(X_train_actual):,} samples with stratified sampling")
        except ValueError as e:
            # If stratified sampling fails (e.g., some classes too small), use random sampling
            print(f"  [SAMPLING] Stratified sampling failed, using random sampling: {e}")
            np.random.seed(42)
            indices = np.random.choice(original_size, size=min(sample_size, original_size), replace=False)
            X_train_actual = X_train[indices]
            y_train_actual = y_train[indices]
            print(f"  [SAMPLING] Randomly sampled {len(X_train_actual):,} samples")
    
    cv_scores = cross_val_score(
        model, X_train_actual, y_train_actual,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1
    )
    
    print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return cv_scores.mean(), cv_scores.std()


def evaluate_model(model, X_val, y_val, model_name):
    """Evaluate model on validation set."""
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average='macro')
    f1_weighted = f1_score(y_val, y_pred, average='weighted')
    
    return {
        'accuracy': float(acc),
        'macro_f1': float(f1_macro),
        'weighted_f1': float(f1_weighted)
    }


def main():
    print("=" * 100)
    print("HYPERPARAMETER TUNING: ALL MODELS")
    print("=" * 100)
    
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
        X_train_full = data['X_train']
        X_val_full = data['X_val']
        X_test_full = data['X_test']
        y_train = data['y_train']
        y_val = data['y_val']
        y_test = data['y_test']
        feature_names = data.get('feature_names', X_train_full.columns.tolist())
    else:
        X_train_full = data['train']
        X_val_full = data['val']
        X_test_full = data['test']
        y_train = data['y_train']
        y_val = data['y_val']
        y_test = data['y_test']
        feature_names = X_train_full.columns.tolist()
    
    print(f"Full features: {len(feature_names)}")
    
    # Handle 5-class severity labels
    unique_levels = sorted(y_train.unique())
    if set(unique_levels).issubset({1, 2, 3, 4, 5}) and len(unique_levels) == 5:
        print(f"\nRemapping 5-class severity to 0-based: 1->0, 2->1, 3->2, 4->3, 5->4")
        y_train = (y_train - 1).astype(np.int32)
        y_val = (y_val - 1).astype(np.int32)
        y_test = (y_test - 1).astype(np.int32)
    
    # Convert to numpy arrays
    X_train_full_array = X_train_full.values if isinstance(X_train_full, pd.DataFrame) else X_train_full
    X_val_full_array = X_val_full.values if isinstance(X_val_full, pd.DataFrame) else X_val_full
    
    # Filter to Age + RFV + Vitals
    X_train_52, X_val_52, _, selected_52 = filter_age_rfv_vitals_features(
        X_train_full, X_val_full, X_test_full, feature_names
    )
    X_train_52_array = X_train_52.values if isinstance(X_train_52, pd.DataFrame) else X_train_52
    X_val_52_array = X_val_52.values if isinstance(X_val_52, pd.DataFrame) else X_val_52
    
    print(f"\nFeature sets:")
    print(f"  Full Features: {X_train_full_array.shape[1]} features")
    print(f"  Age+RFV+Vitals: {X_train_52_array.shape[1]} features")
    
    # Get parameter grids
    param_grids = get_parameter_grids()
    models = ['LogisticRegression', 'RandomForest', 'XGBoost', 'LightGBM']
    feature_sets = {
        'Full_72': (X_train_full_array, X_val_full_array, len(feature_names)),
        'Age_RFV_Vitals_52': (X_train_52_array, X_val_52_array, len(selected_52))
    }
    
    # Store results
    all_results = {}
    
    # RandomizedSearchCV for all models
    print("\n" + "=" * 100)
    print("RANDOMIZEDSEARCHCV FOR ALL MODELS (3-fold CV)")
    print("=" * 100)
    
    for feature_set_name, (X_train_set, X_val_set, n_features) in feature_sets.items():
        print(f"\n{'='*100}")
        print(f"FEATURE SET: {feature_set_name} ({n_features} features)")
        print(f"{'='*100}")
        
        all_results[feature_set_name] = {}
        
        for model_name in models:
            print(f"\n{'-'*100}")
            print(f"MODEL: {model_name}")
            print(f"{'-'*100}")
            
            param_grid = param_grids[model_name].copy()
            
            # Run RandomizedSearchCV
            random_search, search_time = run_randomized_search(
                model_name, X_train_set, y_train, param_grid, n_iter=50, cv=3
            )
            
            # Evaluate best model
            best_model = random_search.best_estimator_
            val_results = evaluate_model(best_model, X_val_set, y_val, model_name)
            
            # Final 3-fold validation
            cv_mean, cv_std = final_validation(best_model, model_name, X_train_set, y_train, cv=3)
            
            # Store results
            all_results[feature_set_name][model_name] = {
                'randomized_search': {
                    'best_score': float(random_search.best_score_),
                    'best_params': {k: (v if not isinstance(v, np.integer) else int(v)) 
                                   for k, v in random_search.best_params_.items()},
                    'search_time': search_time,
                    'n_iter': 50
                },
                'validation': {
                    'accuracy': val_results['accuracy'],
                    'macro_f1': val_results['macro_f1'],
                    'weighted_f1': val_results['weighted_f1'],
                    'cv_3fold_mean': float(cv_mean),
                    'cv_3fold_std': float(cv_std)
                },
                'best_model': best_model
            }
            
            # Save best model
            models_dir = project_root / "services" / "manage-agent" / "models"
            models_dir.mkdir(exist_ok=True)
            model_file = models_dir / f"best_{model_name}_{feature_set_name}.pkl"
            joblib.dump(best_model, model_file)
            print(f"  [OK] Saved: {model_file}")
    
    # Save results
    output_dir = project_root / "services" / "manage-agent" / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Remove model objects from results for JSON serialization
    results_json = {}
    for feature_set, models_dict in all_results.items():
        results_json[feature_set] = {}
        for model_name, model_results in models_dict.items():
            results_json[feature_set][model_name] = {
                k: v for k, v in model_results.items() if k != 'best_model'
            }
    
    results_file = output_dir / "hyperparameter_tuning_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\n[OK] Results saved: {results_file}")
    
    # Print summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    for feature_set_name, models_dict in all_results.items():
        print(f"\n{feature_set_name}:")
        print(f"{'Model':<25} {'CV Score':<12} {'Val Accuracy':<15} {'Macro F1':<12}")
        print("-" * 70)
        for model_name, results in models_dict.items():
            # All models use RandomizedSearchCV
            cv_score = results['randomized_search']['best_score']
            val_acc = results['validation']['accuracy']
            val_f1 = results['validation']['macro_f1']
            print(f"{model_name:<25} {cv_score:<12.4f} {val_acc:<15.4f} {val_f1:<12.4f}")
    
    print("\n" + "=" * 100)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()

