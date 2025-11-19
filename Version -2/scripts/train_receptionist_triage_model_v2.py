"""
Enhanced Receptionist Triage Model Training Pipeline v2

Trains multiple machine learning models with hyperparameter tuning across
4 feature set versions to maximize ESI 1-2 recall.

Feature Sets:
- Version A: Original 16 features
- Version B: Original + age bins (21 features)
- Version C: Original + age bins + RFV risk groups (24 features)
- Version D: Original + age bins + RFV risk groups + interactions (28 features)

Models:
- Logistic Regression (with hyperparameter tuning)
- Decision Tree (with hyperparameter tuning)
- Random Forest (with hyperparameter tuning)
- XGBoost (with enhanced hyperparameter tuning)

Focus: ESI 1-2 recall >= 75% (clinical priority)
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Import feature engineering module
from feature_engineering import build_feature_set
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = ROOT_DIR / "data" / "receptionist_models_v2"

# Create output directories
for subdir in ["models", "metrics", "parameters", "analysis", "reports", "experiments"]:
    (OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

# Add file handler after directories are created
file_handler = logging.FileHandler(OUTPUT_DIR / "reports" / "training_v2.log")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)


# ============================================================================
# PHASE 1: DATA PREPARATION
# ============================================================================


def load_data() -> tuple:
    """Load processed datasets."""
    logger.info("=" * 80)
    logger.info("PHASE 1: DATA PREPARATION")
    logger.info("=" * 80)

    logger.info("\nLoading processed datasets...")
    X_train = pd.read_csv(DATA_DIR / "X_train_final.csv")
    X_val = pd.read_csv(DATA_DIR / "X_val_final.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test_final.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
    y_val = pd.read_csv(DATA_DIR / "y_val.csv").squeeze()
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

    logger.info("Loaded datasets:")
    logger.info(f"  X_train: {X_train.shape}")
    logger.info(f"  X_val: {X_val.shape}")
    logger.info(f"  X_test: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_feature_engineering(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, version: str
) -> tuple:
    """Apply feature engineering based on version."""
    logger.info(f"\nApplying feature engineering for version {version}...")

    X_train_fe = build_feature_set(X_train, version=version)
    X_val_fe = build_feature_set(X_val, version=version)
    X_test_fe = build_feature_set(X_test, version=version)

    # Fix RFV clusters: rows with all zeros should be assigned to "Other" cluster
    rfv_cols = [col for col in X_train_fe.columns if col.startswith("rfv1_cluster_")]
    if rfv_cols:
        for name, df_set in [
            ("Train", X_train_fe),
            ("Val", X_val_fe),
            ("Test", X_test_fe),
        ]:
            rfv_sum = df_set[rfv_cols].sum(axis=1)
            zero_rows = (rfv_sum == 0).sum()
            if zero_rows > 0:
                logger.info(
                    f"  {name}: {zero_rows:,} rows with all zeros in RFV clusters"
                )
                if "rfv1_cluster_Other" in df_set.columns:
                    df_set.loc[rfv_sum == 0, "rfv1_cluster_Other"] = 1.0
                else:
                    most_common_cluster = df_set[rfv_cols].sum().idxmax()
                    df_set.loc[rfv_sum == 0, most_common_cluster] = 1.0

    logger.info(f"Feature set {version}: {X_train_fe.shape[1]} features")
    logger.info(f"  Features: {list(X_train_fe.columns)}")

    return X_train_fe, X_val_fe, X_test_fe


def analyze_esi_distribution(y_train: pd.Series) -> dict:
    """Analyze ESI class distribution."""
    logger.info("\nESI Distribution in Training Set:")
    esi_dist = y_train.value_counts().sort_index()
    total = len(y_train)

    distribution = {}
    for esi in [1, 2, 3, 4, 5]:
        count = esi_dist.get(esi, 0)
        pct = (count / total * 100) if total > 0 else 0
        distribution[esi] = {"count": int(count), "percentage": float(pct)}
        logger.info(f"  ESI {esi}: {count:,} ({pct:.2f}%)")

    return distribution


def apply_enhanced_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    esi_distribution: dict,
    enhanced: bool = True,
) -> tuple:
    """Apply SMOTE + Tomek Links with enhanced strategy."""
    logger.info("\n" + "=" * 80)
    logger.info("APPLYING ENHANCED SMOTE + TOMEK LINKS")
    logger.info("=" * 80)

    if enhanced:
        # Enhanced strategy: more aggressive for ESI 1-2
        # Note: SMOTE only supports oversampling, so we keep original
        # counts for classes that would be undersampled
        sampling_strategy = {
            1: 20000,  # ESI 1: oversample to 20k (was 10k)
            2: 30000,  # ESI 2: oversample to 30k (was 25k)
            3: esi_distribution[3][
                "count"
            ],  # ESI 3: keep original (54k) - SMOTE doesn't undersample
            4: esi_distribution[4][
                "count"
            ],  # ESI 4: keep original (38k) - SMOTE doesn't undersample
            5: 15000,  # ESI 5: oversample to 15k (was 10k)
        }
    else:
        # Original strategy
        sampling_strategy = {
            1: 10000,
            2: 25000,
            3: esi_distribution[3]["count"],
            4: esi_distribution[4]["count"],
            5: 10000,
        }

    logger.info("\nSMOTE sampling strategy:")
    for esi, target_count in sampling_strategy.items():
        current = esi_distribution[esi]["count"]
        logger.info(f"  ESI {esi}: {current:,} -> {target_count:,}")

    logger.info("\nFitting SMOTETomek...")
    sampler = SMOTETomek(sampling_strategy=sampling_strategy, random_state=42)

    start_time = time.time()
    X_train_balanced, y_train_balanced = sampler.fit_resample(X_train, y_train)
    smote_time = time.time() - start_time

    logger.info(f"SMOTE completed in {smote_time:.2f} seconds")
    logger.info("\nBalanced dataset:")
    logger.info(f"  X_train_balanced: {X_train_balanced.shape}")
    logger.info(f"  y_train_balanced: {len(y_train_balanced)}")

    # Verify balanced distribution
    logger.info("\nBalanced ESI Distribution:")
    balanced_dist = pd.Series(y_train_balanced).value_counts().sort_index()
    for esi in [1, 2, 3, 4, 5]:
        count = balanced_dist.get(esi, 0)
        pct = count / len(y_train_balanced) * 100
        logger.info(f"  ESI {esi}: {count:,} ({pct:.2f}%)")

    return X_train_balanced, y_train_balanced


# ============================================================================
# PHASE 2: HYPERPARAMETER TUNING FUNCTIONS
# ============================================================================


def tune_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series, cv: int = 3, sample_size: int = 10000
) -> tuple:
    """Tune Logistic Regression with GridSearchCV on sampled data for speed."""
    logger.info("\n" + "=" * 80)
    logger.info("TUNING LOGISTIC REGRESSION")
    logger.info("=" * 80)

    # Sample for grid search to speed up hyperparameter tuning
    if len(X_train) > sample_size:
        logger.info(
            f"Sampling {sample_size:,} rows from {len(X_train):,} for grid search..."
        )
        X_train_sample, _, y_train_sample, _ = train_test_split(
            X_train, y_train, train_size=sample_size, stratify=y_train, random_state=42
        )
        logger.info(f"Sample shape: {X_train_sample.shape}")
    else:
        X_train_sample = X_train
        y_train_sample = y_train
        logger.info(f"Using full dataset ({len(X_train):,} rows) for grid search")

    # Optimized parameter grid - only 10 combinations (5 C x 1 penalty x 2 class_weight)
    # Using GridSearchCV for exhaustive search since space is small
    param_grid = {
        "C": [0.1, 1.0, 10.0, 100.0, 1000.0],  # Keep higher C values for convergence
        "penalty": ["l2"],  # Focus on L2 (most common, faster than elasticnet)
        "solver": ["saga"],  # saga solver handles multiclass well
        "class_weight": [
            "balanced",
            {1: 50, 2: 25, 3: 1, 4: 1, 5: 5},
        ],
    }

    # Create base model with lower max_iter for CV (faster),
    # will use higher for final model
    base_model = LogisticRegression(
        solver="saga",
        max_iter=5000,  # Lower for CV speed, will increase for final model
        tol=1e-4,
        random_state=42,
        n_jobs=1,  # saga doesn't support n_jobs > 1, but CV can parallelize
    )

    # Use GridSearchCV with parallelization - only 10 combinations so exhaustive is fine
    cv_fold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv_fold,
        scoring="f1_macro",
        n_jobs=-1,  # Parallelize across CV folds and parameter combinations
        verbose=1,
        error_score="raise",
    )

    logger.info(
        f"Running grid search on sampled data "
        f"(10 combinations, {cv}-fold CV, parallelized)..."
    )
    start_time = time.time()

    # Suppress convergence warnings during search
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, message=".*did not converge.*"
        )
        grid_search.fit(X_train_sample, y_train_sample)

    elapsed_time = time.time() - start_time
    logger.info(f"Grid search completed in {elapsed_time:.2f} seconds")

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    logger.info(f"\nBest parameters: {best_params}")
    logger.info(f"Best CV score (f1_macro) on sample: {best_score:.4f}")

    # Train final model on full training set with higher max_iter
    logger.info(f"\nTraining final model on full dataset ({len(X_train):,} rows)...")
    final_model = LogisticRegression(
        **best_params,
        max_iter=10000,  # Higher for final model to ensure convergence
        tol=1e-4,
        random_state=42,
    )

    # Suppress convergence warnings for final model
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, message=".*did not converge.*"
        )
        final_model.fit(X_train, y_train)

    # Log if convergence was reached
    if hasattr(final_model, "n_iter_") and final_model.n_iter_[0] >= 10000:
        logger.warning("Final model reached max_iter - may not have fully converged")
    else:
        logger.info(f"Final model converged in {final_model.n_iter_[0]} iterations")

    # Convert CV results to DataFrame
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    cv_results_df = cv_results_df.sort_values("mean_test_score", ascending=False)
    # Select relevant columns
    cv_results_df = cv_results_df[
        [
            "param_C",
            "param_penalty",
            "param_class_weight",
            "mean_test_score",
            "std_test_score",
        ]
    ].rename(
        columns={
            "param_C": "C",
            "param_penalty": "penalty",
            "param_class_weight": "class_weight",
            "mean_test_score": "mean_score",
            "std_test_score": "std_score",
        }
    )

    return final_model, best_params, cv_results_df


def tune_decision_tree(X_train: pd.DataFrame, y_train: pd.Series, cv: int = 3) -> tuple:
    """Tune Decision Tree with GridSearchCV."""
    logger.info("\n" + "=" * 80)
    logger.info("TUNING DECISION TREE")
    logger.info("=" * 80)

    param_grid = {
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "class_weight": ["balanced", {1: 50, 2: 25, 3: 1, 4: 1, 5: 5}],
        "criterion": ["gini", "entropy"],
    }

    base_model = DecisionTreeClassifier(random_state=42)

    cv_fold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv_fold,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )

    logger.info("Running grid search (3-fold CV)...")
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    grid_time = time.time() - start_time

    logger.info(f"Grid search completed in {grid_time:.2f} seconds")
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best CV score (f1_macro): {grid_search.best_score_:.4f}")

    # Train final model on full training set
    final_model = DecisionTreeClassifier(**grid_search.best_params_, random_state=42)
    final_model.fit(X_train, y_train)

    cv_results_df = pd.DataFrame(grid_search.cv_results_)

    return final_model, grid_search.best_params_, cv_results_df


def tune_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series, cv: int = 3, n_iter: int = 50
) -> tuple:
    """Tune Random Forest with RandomizedSearchCV."""
    logger.info("\n" + "=" * 80)
    logger.info("TUNING RANDOM FOREST")
    logger.info("=" * 80)

    param_distributions = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2", 0.5],
        "class_weight": ["balanced", {1: 50, 2: 25, 3: 1, 4: 1, 5: 5}],
    }

    base_model = RandomForestClassifier(oob_score=True, random_state=42, n_jobs=-1)

    cv_fold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(
        base_model,
        param_distributions,
        n_iter=n_iter,
        cv=cv_fold,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )

    logger.info(f"Running random search ({n_iter} iterations, 3-fold CV)...")
    start_time = time.time()
    random_search.fit(X_train, y_train)
    search_time = time.time() - start_time

    logger.info(f"Random search completed in {search_time:.2f} seconds")
    logger.info(f"Best parameters: {random_search.best_params_}")
    logger.info(f"Best CV score (f1_macro): {random_search.best_score_:.4f}")

    # Train final model on full training set
    final_model = RandomForestClassifier(
        **random_search.best_params_, oob_score=True, random_state=42, n_jobs=-1
    )
    final_model.fit(X_train, y_train)

    cv_results_df = pd.DataFrame(random_search.cv_results_)

    return final_model, random_search.best_params_, cv_results_df


def tune_xgboost_enhanced(
    X_train: pd.DataFrame, y_train: pd.Series, cv: int = 3, sample_size: int = 5000
) -> tuple:
    """Tune XGBoost with enhanced parameter grid on sampled data for speed."""
    logger.info("\n" + "=" * 80)
    logger.info("TUNING XGBOOST (ENHANCED)")
    logger.info("=" * 80)

    # XGBoost requires class labels to start from 0
    logger.info("Converting ESI labels from 1-5 to 0-4 for XGBoost...")
    y_train_xgb = y_train - 1

    # Sample for grid search
    if len(X_train) > sample_size:
        logger.info(
            f"Sampling {sample_size:,} rows from {len(X_train):,} for grid search..."
        )
        X_train_sample, _, y_train_sample, _ = train_test_split(
            X_train,
            y_train_xgb,
            train_size=sample_size,
            stratify=y_train_xgb,
            random_state=42,
        )
    else:
        X_train_sample = X_train
        y_train_sample = y_train_xgb

    # Enhanced parameter grid
    param_grid = {
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [200, 300, 500],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_alpha": [0, 0.1, 0.5, 1.0],
        "reg_lambda": [0.1, 1.0, 5.0],
        "gamma": [0, 0.1, 0.5],
    }

    base_model = XGBClassifier(
        objective="multi:softmax", num_class=5, random_state=42, n_jobs=-1
    )

    cv_fold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv_fold,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )

    logger.info(
        f"Running grid search on sampled data ({sample_size:,} rows, {cv}-fold CV)..."
    )
    start_time = time.time()
    grid_search.fit(X_train_sample, y_train_sample)
    grid_time = time.time() - start_time

    logger.info(f"Grid search completed in {grid_time:.2f} seconds")
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best CV score (f1_macro) on sample: {grid_search.best_score_:.4f}")

    # Train final model on full training set with aggressive cost weights
    logger.info(
        f"\nTraining final XGBoost model on full dataset "
        f"({len(X_train):,} rows) with cost-sensitive weights..."
    )
    best_params = grid_search.best_params_.copy()
    class_weights_xgb = {0: 100, 1: 50, 2: 1, 3: 1, 4: 10}  # ESI 1->0, 2->1, etc.
    sample_weights_xgb = np.array([class_weights_xgb[int(y)] for y in y_train_xgb])

    final_model = XGBClassifier(
        **best_params,
        objective="multi:softmax",
        num_class=5,
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X_train, y_train_xgb, sample_weight=sample_weights_xgb)

    cv_results_df = pd.DataFrame(grid_search.cv_results_)

    return final_model, best_params, cv_results_df


# ============================================================================
# PHASE 3: EVALUATION METRICS
# ============================================================================


def calculate_metrics_enhanced(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray | None,
    model_name: str,
) -> dict:
    """Calculate comprehensive evaluation metrics with ESI 1-2 focus."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[1, 2, 3, 4, 5], zero_division=0
    )

    # Per-ESI metrics
    metrics_dict = {
        "model": model_name,
        "accuracy": float(accuracy),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }

    # ESI 1-2 combined recall (PRIMARY METRIC)
    esi_1_2_mask = y_true.isin([1, 2])
    if esi_1_2_mask.sum() > 0:
        esi_1_2_true = y_true[esi_1_2_mask]
        esi_1_2_pred = y_pred[esi_1_2_mask]
        tp_esi1 = ((esi_1_2_true == 1) & (esi_1_2_pred == 1)).sum()
        tp_esi2 = ((esi_1_2_true == 2) & (esi_1_2_pred == 2)).sum()
        total_esi1_2 = len(esi_1_2_true)
        esi_1_2_recall = (tp_esi1 + tp_esi2) / total_esi1_2 if total_esi1_2 > 0 else 0.0
        metrics_dict["esi_1_2_recall"] = float(esi_1_2_recall)
    else:
        metrics_dict["esi_1_2_recall"] = 0.0

    # ESI 1 recall
    esi_1_mask = y_true == 1
    if esi_1_mask.sum() > 0:
        esi_1_recall = recall[0]  # ESI 1 is first in labels=[1,2,3,4,5]
        metrics_dict["esi_1_recall"] = float(esi_1_recall)
    else:
        metrics_dict["esi_1_recall"] = 0.0

    # ESI 2 recall
    esi_2_mask = y_true == 2
    if esi_2_mask.sum() > 0:
        esi_2_recall = recall[1]  # ESI 2 is second
        metrics_dict["esi_2_recall"] = float(esi_2_recall)
    else:
        metrics_dict["esi_2_recall"] = 0.0

    # Per-ESI metrics
    for i, esi in enumerate([1, 2, 3, 4, 5]):
        idx = i
        metrics_dict[f"esi_{esi}_precision"] = float(precision[idx])
        metrics_dict[f"esi_{esi}_recall"] = float(recall[idx])
        metrics_dict[f"esi_{esi}_f1"] = float(f1[idx])
        metrics_dict[f"esi_{esi}_support"] = int(support[idx])

    return metrics_dict


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    feature_set: str,
    output_dir: Path,
) -> None:
    """Save confusion matrix to file."""
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])
    cm_df = pd.DataFrame(cm, index=[1, 2, 3, 4, 5], columns=[1, 2, 3, 4, 5])
    filename = (
        f"confusion_matrix_{model_name.lower().replace(' ', '_')}_{feature_set}.csv"
    )
    cm_df.to_csv(output_dir / "analysis" / filename)
    logger.info(f"Saved confusion matrix to: {output_dir / 'analysis' / filename}")


# ============================================================================
# PHASE 4: TRAINING PIPELINE
# ============================================================================


def train_and_evaluate_model(
    model_name: str,
    tune_func,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_set: str,
    output_dir: Path,
) -> tuple:
    """Train and evaluate a single model."""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"TRAINING {model_name.upper()} (Feature Set {feature_set})")
    logger.info(f"{'=' * 80}")

    # Tune hyperparameters
    start_time = time.time()
    model, best_params, cv_results = tune_func(X_train, y_train)
    tune_time = time.time() - start_time

    logger.info(f"Hyperparameter tuning completed in {tune_time:.2f} seconds")

    # Save CV results
    cv_results_file = (
        output_dir
        / "parameters"
        / f"cv_results_{model_name.lower().replace(' ', '_')}_{feature_set}.csv"
    )
    cv_results.to_csv(cv_results_file, index=False)
    logger.info(f"Saved CV results to: {cv_results_file}")

    # Save best parameters
    params_file = (
        output_dir
        / "parameters"
        / f"best_params_{model_name.lower().replace(' ', '_')}_{feature_set}.json"
    )
    with open(params_file, "w") as f:
        json.dump(best_params, f, indent=2)
    logger.info(f"Saved best parameters to: {params_file}")

    # Evaluate on validation set
    logger.info("\nEvaluating on validation set...")
    y_pred = model.predict(X_val)

    # Handle XGBoost label conversion
    if isinstance(model, XGBClassifier):
        y_pred = y_pred + 1  # Convert back to 1-5

    y_pred_proba = None
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_val)
        if isinstance(model, XGBClassifier):
            # XGBoost probabilities are already in correct shape
            pass

    metrics = calculate_metrics_enhanced(
        y_val, y_pred, y_pred_proba, f"{model_name} ({feature_set})"
    )

    # Save confusion matrix
    save_confusion_matrix(y_val, y_pred, model_name, feature_set, output_dir)

    # Save metrics
    metrics_file = (
        output_dir
        / "metrics"
        / f"val_metrics_{model_name.lower().replace(' ', '_')}_{feature_set}.csv"
    )
    pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
    logger.info(f"Saved metrics to: {metrics_file}")

    # Save model
    model_file = (
        output_dir
        / "models"
        / f"{model_name.lower().replace(' ', '_')}_{feature_set}.pkl"
    )
    joblib.dump(model, model_file)
    logger.info(f"Saved model to: {model_file}")

    return model, metrics, best_params


def main():
    """Main training pipeline."""
    logger.info("=" * 80)
    logger.info("ENHANCED RECEPTIONIST TRIAGE MODEL TRAINING v2")
    logger.info("=" * 80)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # Analyze ESI distribution
    esi_distribution = analyze_esi_distribution(y_train)

    # Define feature sets and models
    feature_sets = ["A", "B", "C", "D"]
    models_config = {
        "Logistic Regression": tune_logistic_regression,
        "Decision Tree": tune_decision_tree,
        "Random Forest": tune_random_forest,
        "XGBoost": tune_xgboost_enhanced,
    }

    # Store all results
    all_results = {}

    # Loop over feature sets
    for feature_set in feature_sets:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"FEATURE SET {feature_set}")
        logger.info(f"{'=' * 80}")

        # Apply feature engineering
        X_train_fe, X_val_fe, X_test_fe = apply_feature_engineering(
            X_train, X_val, X_test, feature_set
        )

        # Apply enhanced SMOTE
        X_train_balanced, y_train_balanced = apply_enhanced_smote(
            X_train_fe, y_train, esi_distribution, enhanced=True
        )

        # Convert to DataFrame if needed
        if isinstance(X_train_balanced, np.ndarray):
            X_train_balanced = pd.DataFrame(
                X_train_balanced, columns=X_train_fe.columns
            )
        if isinstance(y_train_balanced, np.ndarray):
            y_train_balanced = pd.Series(y_train_balanced)

        # Train all models
        feature_set_results = {}
        for model_name, tune_func in models_config.items():
            try:
                model, metrics, best_params = train_and_evaluate_model(
                    model_name,
                    tune_func,
                    X_train_balanced,
                    y_train_balanced,
                    X_val_fe,
                    y_val,
                    feature_set,
                    OUTPUT_DIR,
                )
                feature_set_results[model_name] = {
                    "model": model,
                    "metrics": metrics,
                    "params": best_params,
                }
            except Exception as e:
                logger.error(
                    f"Error training {model_name} on feature set {feature_set}: {e}"
                )
                continue

        all_results[feature_set] = feature_set_results

        # Select best model for this feature set
        if feature_set_results:
            best_model_name = max(
                feature_set_results.keys(),
                key=lambda k: feature_set_results[k]["metrics"].get(
                    "esi_1_2_recall", 0
                ),
            )
            logger.info(
                f"\nBest model for feature set {feature_set}: {best_model_name}"
            )
            metrics = feature_set_results[best_model_name]["metrics"]
            esi_recall = metrics["esi_1_2_recall"]
            logger.info(f"ESI 1-2 Recall: {esi_recall:.4f}")

    # Evaluate best models on test set
    logger.info(f"\n{'=' * 80}")
    logger.info("EVALUATING BEST MODELS ON TEST SET")
    logger.info(f"{'=' * 80}")

    test_results = {}
    for feature_set in feature_sets:
        if feature_set not in all_results or not all_results[feature_set]:
            continue

        # Get best model for this feature set
        best_model_name = max(
            all_results[feature_set].keys(),
            key=lambda k: all_results[feature_set][k]["metrics"].get(
                "esi_1_2_recall", 0
            ),
        )

        # Apply feature engineering to test set
        _, _, X_test_fe = apply_feature_engineering(X_train, X_val, X_test, feature_set)

        # Load model
        model_file = (
            OUTPUT_DIR
            / "models"
            / f"{best_model_name.lower().replace(' ', '_')}_{feature_set}.pkl"
        )
        if not model_file.exists():
            logger.warning(f"Model file not found: {model_file}")
            continue

        model = joblib.load(model_file)
        logger.info(
            f"\nEvaluating {best_model_name} (Feature Set {feature_set}) on test set..."
        )

        # Make predictions
        y_pred = model.predict(X_test_fe)
        if isinstance(model, XGBClassifier):
            y_pred = y_pred + 1  # Convert back to 1-5

        y_pred_proba = None
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test_fe)

        # Calculate metrics
        test_metrics = calculate_metrics_enhanced(
            y_test, y_pred, y_pred_proba, f"{best_model_name} ({feature_set}) - Test"
        )

        # Save confusion matrix
        save_confusion_matrix(
            y_test, y_pred, best_model_name, f"{feature_set}_test", OUTPUT_DIR
        )

        # Save test metrics
        model_name_safe = best_model_name.lower().replace(" ", "_")
        test_metrics_file = (
            OUTPUT_DIR / "metrics" / f"test_metrics_{model_name_safe}_{feature_set}.csv"
        )
        pd.DataFrame([test_metrics]).to_csv(test_metrics_file, index=False)

        test_results[feature_set] = {
            "model": best_model_name,
            "metrics": test_metrics,
        }

        logger.info(f"Test ESI 1-2 Recall: {test_metrics['esi_1_2_recall']:.4f}")

    # Save experiment summary
    experiment_summary = {
        "feature_sets": feature_sets,
        "models": list(models_config.keys()),
        "validation_results": {
            fs: {
                model: {
                    "esi_1_2_recall": results["metrics"]["esi_1_2_recall"],
                    "accuracy": results["metrics"]["accuracy"],
                    "macro_f1": results["metrics"]["macro_f1"],
                }
                for model, results in fs_results.items()
            }
            for fs, fs_results in all_results.items()
        },
        "test_results": {
            fs: {
                "model": results["model"],
                "esi_1_2_recall": results["metrics"]["esi_1_2_recall"],
                "accuracy": results["metrics"]["accuracy"],
                "macro_f1": results["metrics"]["macro_f1"],
            }
            for fs, results in test_results.items()
        },
    }

    summary_file = OUTPUT_DIR / "experiments" / "experiment_summary.json"
    with open(summary_file, "w") as f:
        json.dump(experiment_summary, f, indent=2)
    logger.info(f"\nSaved experiment summary to: {summary_file}")

    # Generate markdown report
    logger.info("\nGenerating experiment summary report...")
    report_lines = []
    report_lines.append("# Experiment Summary Report")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("## Overview")
    report_lines.append("")
    report_lines.append(f"- Feature Sets Evaluated: {', '.join(feature_sets)}")
    report_lines.append(f"- Models Evaluated: {', '.join(models_config.keys())}")
    report_lines.append("")

    # Best models per feature set
    report_lines.append("## Best Model Per Feature Set (Test Set)")
    report_lines.append("")
    for feature_set in feature_sets:
        if feature_set in test_results:
            result = test_results[feature_set]
            metrics = result["metrics"]
            report_lines.append(f"### Feature Set {feature_set}")
            report_lines.append(f"- **Model**: {result['model']}")
            report_lines.append(
                f"- **ESI 1-2 Recall**: {metrics.get('esi_1_2_recall', 0):.4f}"
            )
            report_lines.append(f"- **Accuracy**: {metrics.get('accuracy', 0):.4f}")
            report_lines.append(f"- **Macro F1**: {metrics.get('macro_f1', 0):.4f}")
            report_lines.append("")

    # Overall best
    if test_results:
        best_overall = max(
            test_results.items(), key=lambda x: x[1]["metrics"].get("esi_1_2_recall", 0)
        )
        best_fs, best_result = best_overall
        best_metrics = best_result["metrics"]
        report_lines.append("## Best Overall Model (Test Set)")
        report_lines.append("")
        report_lines.append(f"- **Model**: {best_result['model']}")
        report_lines.append(f"- **Feature Set**: {best_fs}")
        report_lines.append(
            f"- **ESI 1-2 Recall**: {best_metrics.get('esi_1_2_recall', 0):.4f}"
        )
        report_lines.append(f"- **Accuracy**: {best_metrics.get('accuracy', 0):.4f}")
        report_lines.append(f"- **Macro F1**: {best_metrics.get('macro_f1', 0):.4f}")
        report_lines.append("")

        # Recommendations
        esi_recall = best_metrics.get("esi_1_2_recall", 0)
        report_lines.append("## Recommendations")
        report_lines.append("")
        if esi_recall >= 0.75:
            report_lines.append(
                f"- [OK] ESI 1-2 recall ({esi_recall:.2%}) meets "
                "target threshold (>= 75%)"
            )
        elif esi_recall >= 0.50:
            report_lines.append(
                f"- [WARN] ESI 1-2 recall ({esi_recall:.2%}) is below "
                "target but above 50%"
            )
        else:
            report_lines.append(
                f"- [ERROR] ESI 1-2 recall ({esi_recall:.2%}) is below "
                "50% - needs improvement"
            )
        model_name = best_result["model"]
        report_lines.append(
            f"- **Recommended Model**: {model_name} with Feature Set {best_fs}"
        )
        report_lines.append("")

    report_file = OUTPUT_DIR / "reports" / "experiment_summary.md"
    with open(report_file, "w") as f:
        f.write("\n".join(report_lines))
    logger.info(f"Saved experiment summary report to: {report_file}")

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("1. Review experiment_summary.json and experiment_summary.md")
    logger.info("2. Run compare_experiments.py for detailed comparison")
    logger.info("3. Select best model based on ESI 1-2 recall")


if __name__ == "__main__":
    main()
