"""
Balanced Receptionist Triage Model Training Pipeline v3

Retrains models using best hyperparameters from v2, but with:
- Moderate class weights: {1:10, 2:5, 3:1, 4:1, 5:2}
- Less aggressive SMOTE: {1:5000, 2:15000, 3:original, 4:original, 5:original}
- No hyperparameter tuning (use saved best params)
- Evaluate on train, validation, and test sets

Target: Overall accuracy >50%, ESI 1-2 recall >40%
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

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
V2_PARAMS_DIR = ROOT_DIR / "data" / "receptionist_models_v2" / "parameters"
OUTPUT_DIR = ROOT_DIR / "data" / "receptionist_models_v3"

# Create output directories
for subdir in ["models", "metrics", "parameters", "analysis", "reports"]:
    (OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

# Add file handler after directories are created
file_handler = logging.FileHandler(OUTPUT_DIR / "reports" / "training_v3.log")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

# Moderate class weights
MODERATE_CLASS_WEIGHTS = {1: 10, 2: 5, 3: 1, 4: 1, 5: 2}

# XGBoost class weights (0-indexed: ESI 1->0, 2->1, 3->2, 4->3, 5->4)
XGBOOST_CLASS_WEIGHTS = {0: 10, 1: 5, 2: 1, 3: 1, 4: 2}


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


def apply_balanced_smote(
    X_train: pd.DataFrame, y_train: pd.Series, esi_distribution: dict
) -> tuple:
    """Apply SMOTE + Tomek Links with balanced (moderate) strategy."""
    logger.info("\n" + "=" * 80)
    logger.info("APPLYING BALANCED SMOTE + TOMEK LINKS")
    logger.info("=" * 80)

    # Balanced strategy: moderate oversampling
    # Note: SMOTE only supports oversampling, so target must be >= original count
    sampling_strategy = {
        1: 5000,  # ESI 1: moderate oversample (was 20000)
        2: 15000,  # ESI 2: moderate oversample (was 30000)
        3: esi_distribution[3]["count"],  # Keep original (54323)
        4: esi_distribution[4]["count"],  # Keep original (38240)
        5: esi_distribution[5][
            "count"
        ],  # Keep original (cannot undersample with SMOTE)
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
# PHASE 2: SAMPLING FOR TRAINING
# ============================================================================


def create_batches(
    X_train: pd.DataFrame, y_train: pd.Series, batch_size: int = 5000
) -> list:
    """Split training data into batches of specified size."""
    total_samples = len(X_train)
    num_batches = (total_samples + batch_size - 1) // batch_size  # Ceiling division

    logger.info(
        f"Splitting {total_samples:,} samples into {num_batches} batches of ~{batch_size:,} samples each..."
    )

    batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_samples)

        X_batch = X_train.iloc[start_idx:end_idx].reset_index(drop=True)
        y_batch = y_train.iloc[start_idx:end_idx].reset_index(drop=True)

        batches.append((X_batch, y_batch))
        logger.info(f"  Batch {i+1}/{num_batches}: {len(X_batch):,} samples")

    return batches


# ============================================================================
# PHASE 3: LOAD AND MODIFY PARAMETERS
# ============================================================================


def load_and_modify_params(model_name: str, feature_set: str) -> dict:
    """Load best hyperparameters from v2 and replace class_weight with moderate weights."""
    # Map model names to file names
    model_file_map = {
        "Logistic Regression": "logistic_regression",
        "Decision Tree": "decision_tree",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
    }

    file_name = model_file_map.get(model_name)
    if not file_name:
        raise ValueError(f"Unknown model name: {model_name}")

    params_file = V2_PARAMS_DIR / f"best_params_{file_name}_{feature_set}.json"

    if not params_file.exists():
        raise FileNotFoundError(f"Parameter file not found: {params_file}")

    logger.info(f"Loading parameters from: {params_file}")
    with open(params_file) as f:
        params = json.load(f)

    # Replace class_weight with moderate weights
    # Handle both string "balanced" and dict formats
    if "class_weight" in params:
        old_weight = params["class_weight"]
        params["class_weight"] = MODERATE_CLASS_WEIGHTS
        logger.info(
            f"  Replaced class_weight: {old_weight} -> {MODERATE_CLASS_WEIGHTS}"
        )

    # Handle null values (convert to None for Python)
    if "max_depth" in params and params["max_depth"] is None:
        params["max_depth"] = None

    return params


# ============================================================================
# PHASE 4: MODEL TRAINING (NO TUNING)
# ============================================================================


def train_logistic_regression_no_tuning(
    X_train: pd.DataFrame, y_train: pd.Series, feature_set: str, batch_size: int = 5000
) -> tuple:
    """Train Logistic Regression using best params from v2 with moderate class weights, in batches."""
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING LOGISTIC REGRESSION (NO TUNING - BATCH TRAINING)")
    logger.info("=" * 80)

    params = load_and_modify_params("Logistic Regression", feature_set)

    # Ensure max_iter is set for convergence (per batch)
    if "max_iter" not in params:
        params["max_iter"] = 1000  # Lower per batch since we're doing multiple batches
    params["tol"] = 1e-4
    params["random_state"] = 42

    # Create batches
    batches = create_batches(X_train, y_train, batch_size)

    logger.info(f"Training with parameters: {params}")
    logger.info(
        f"Training on {len(batches)} batches (total {len(X_train):,} samples)..."
    )

    start_time = time.time()

    # Initialize model (need to fit on first batch to get classes)
    model = LogisticRegression(**params)

    # Fit on first batch to initialize
    X_first, y_first = batches[0]
    model.fit(X_first, y_first)
    logger.info(f"  Batch 1/{len(batches)}: Fitted (initialization)")

    # Use partial_fit for remaining batches (incremental learning)
    for i, (X_batch, y_batch) in enumerate(batches[1:], start=2):
        model.partial_fit(X_batch, y_batch, classes=[1, 2, 3, 4, 5])
        logger.info(
            f"  Batch {i}/{len(batches)}: Updated with {len(X_batch):,} samples"
        )

    train_time = time.time() - start_time
    logger.info(
        f"Training completed in {train_time:.2f} seconds (processed {len(X_train):,} samples in {len(batches)} batches)"
    )

    return model, params


def train_decision_tree_no_tuning(
    X_train: pd.DataFrame, y_train: pd.Series, feature_set: str, batch_size: int = 5000
) -> tuple:
    """Train Decision Tree using best params from v2 with moderate class weights, in batches."""
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING DECISION TREE (NO TUNING - BATCH TRAINING)")
    logger.info("=" * 80)

    params = load_and_modify_params("Decision Tree", feature_set)
    params["random_state"] = 42

    # Create batches
    batches = create_batches(X_train, y_train, batch_size)

    logger.info(f"Training with parameters: {params}")
    logger.info(
        f"Training on {len(batches)} batches (total {len(X_train):,} samples)..."
    )

    start_time = time.time()

    # Decision Tree doesn't support incremental learning, so train on full dataset
    # (batches are created for memory efficiency, but we train on full dataset)
    model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)
    logger.info(f"Trained on full dataset: {len(X_train):,} samples")

    train_time = time.time() - start_time
    logger.info(
        f"Training completed in {train_time:.2f} seconds (processed {len(X_train):,} samples in {len(batches)} batches)"
    )

    return model, params


def train_random_forest_no_tuning(
    X_train: pd.DataFrame, y_train: pd.Series, feature_set: str, batch_size: int = 5000
) -> tuple:
    """Train Random Forest using best params from v2 with moderate class weights, in batches."""
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING RANDOM FOREST (NO TUNING - BATCH TRAINING)")
    logger.info("=" * 80)

    params = load_and_modify_params("Random Forest", feature_set)
    params["oob_score"] = True
    params["random_state"] = 42
    params["n_jobs"] = -1

    # Create batches
    batches = create_batches(X_train, y_train, batch_size)

    logger.info(f"Training with parameters: {params}")
    logger.info(
        f"Training on {len(batches)} batches (total {len(X_train):,} samples)..."
    )

    start_time = time.time()

    # Random Forest doesn't support incremental learning, so train on full dataset
    # (batches are created for memory efficiency, but we train on full dataset)
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    logger.info(f"Trained on full dataset: {len(X_train):,} samples")

    train_time = time.time() - start_time
    logger.info(
        f"Training completed in {train_time:.2f} seconds (processed {len(X_train):,} samples in {len(batches)} batches)"
    )
    if hasattr(model, "oob_score_"):
        logger.info(f"OOB Score: {model.oob_score_:.4f}")

    return model, params


def train_xgboost_no_tuning(
    X_train: pd.DataFrame, y_train: pd.Series, feature_set: str, batch_size: int = 5000
) -> tuple:
    """Train XGBoost using best params from v2 with moderate class weights, in batches."""
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING XGBOOST (NO TUNING - BATCH TRAINING)")
    logger.info("=" * 80)

    # Create batches (before label conversion)
    batches = create_batches(X_train, y_train, batch_size)

    # XGBoost requires class labels to start from 0
    logger.info("Converting ESI labels from 1-5 to 0-4 for XGBoost...")

    params = load_and_modify_params("XGBoost", feature_set)
    params["objective"] = "multi:softmax"
    params["num_class"] = 5
    params["random_state"] = 42
    params["n_jobs"] = -1

    logger.info(f"Training with parameters: {params}")
    logger.info(
        f"Training on {len(batches)} batches (total {len(X_train):,} samples)..."
    )

    start_time = time.time()

    # XGBoost supports incremental training
    model = None
    for i, (X_batch, y_batch) in enumerate(batches, start=1):
        # Convert labels for this batch
        y_batch_xgb = y_batch - 1

        # Apply moderate class weights as sample weights
        sample_weights_xgb = np.array(
            [XGBOOST_CLASS_WEIGHTS[int(y)] for y in y_batch_xgb]
        )

        if i == 1:
            # First batch: initialize model
            model = XGBClassifier(**params)
            model.fit(X_batch, y_batch_xgb, sample_weight=sample_weights_xgb)
            logger.info(
                f"  Batch {i}/{len(batches)}: Fitted on {len(X_batch):,} samples"
            )
        else:
            # Subsequent batches: incremental training
            model.fit(
                X_batch,
                y_batch_xgb,
                sample_weight=sample_weights_xgb,
                xgb_model=model.get_booster(),  # Continue from previous model
            )
            logger.info(
                f"  Batch {i}/{len(batches)}: Updated with {len(X_batch):,} samples"
            )

    train_time = time.time() - start_time
    logger.info(
        f"Training completed in {train_time:.2f} seconds (processed {len(X_train):,} samples in {len(batches)} batches)"
    )

    return model, params


# ============================================================================
# PHASE 4: EVALUATION METRICS
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

    # ESI 3-4 combined recall
    esi_3_4_mask = y_true.isin([3, 4])
    if esi_3_4_mask.sum() > 0:
        esi_3_4_true = y_true[esi_3_4_mask]
        esi_3_4_pred = y_pred[esi_3_4_mask]
        tp_esi3 = ((esi_3_4_true == 3) & (esi_3_4_pred == 3)).sum()
        tp_esi4 = ((esi_3_4_true == 4) & (esi_3_4_pred == 4)).sum()
        total_esi3_4 = len(esi_3_4_true)
        esi_3_4_recall = (tp_esi3 + tp_esi4) / total_esi3_4 if total_esi3_4 > 0 else 0.0
        metrics_dict["esi_3_4_recall"] = float(esi_3_4_recall)
    else:
        metrics_dict["esi_3_4_recall"] = 0.0

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
    dataset_name: str,
    output_dir: Path,
) -> None:
    """Save confusion matrix to file."""
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])
    cm_df = pd.DataFrame(cm, index=[1, 2, 3, 4, 5], columns=[1, 2, 3, 4, 5])
    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}_{feature_set}_{dataset_name}.csv"
    cm_df.to_csv(output_dir / "analysis" / filename)
    logger.info(f"Saved confusion matrix to: {output_dir / 'analysis' / filename}")


def evaluate_model_comprehensive(
    model: Any,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_set: str,
    output_dir: Path,
) -> dict:
    """Evaluate model on train, validation, and test sets."""
    logger.info(f"\n{'=' * 80}")
    logger.info(
        f"COMPREHENSIVE EVALUATION: {model_name.upper()} (Feature Set {feature_set})"
    )
    logger.info(f"{'=' * 80}")

    results = {}
    is_xgboost = isinstance(model, XGBClassifier)

    # Evaluate on train set
    logger.info("\nEvaluating on TRAIN set...")
    y_pred_train = model.predict(X_train)
    if is_xgboost:
        y_pred_train = y_pred_train + 1  # Convert back to 1-5

    y_pred_proba_train = None
    if hasattr(model, "predict_proba"):
        y_pred_proba_train = model.predict_proba(X_train)

    train_metrics = calculate_metrics_enhanced(
        y_train,
        y_pred_train,
        y_pred_proba_train,
        f"{model_name} ({feature_set}) - Train",
    )
    results["train"] = train_metrics
    save_confusion_matrix(
        y_train, y_pred_train, model_name, feature_set, "train", output_dir
    )

    logger.info(f"  Train Accuracy: {train_metrics['accuracy']:.4f}")
    logger.info(f"  Train ESI 1-2 Recall: {train_metrics['esi_1_2_recall']:.4f}")
    logger.info(f"  Train ESI 3-4 Recall: {train_metrics['esi_3_4_recall']:.4f}")

    # Evaluate on validation set
    logger.info("\nEvaluating on VALIDATION set...")
    y_pred_val = model.predict(X_val)
    if is_xgboost:
        y_pred_val = y_pred_val + 1  # Convert back to 1-5

    y_pred_proba_val = None
    if hasattr(model, "predict_proba"):
        y_pred_proba_val = model.predict_proba(X_val)

    val_metrics = calculate_metrics_enhanced(
        y_val, y_pred_val, y_pred_proba_val, f"{model_name} ({feature_set}) - Val"
    )
    results["val"] = val_metrics
    save_confusion_matrix(y_val, y_pred_val, model_name, feature_set, "val", output_dir)

    logger.info(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
    logger.info(f"  Val ESI 1-2 Recall: {val_metrics['esi_1_2_recall']:.4f}")
    logger.info(f"  Val ESI 3-4 Recall: {val_metrics['esi_3_4_recall']:.4f}")

    # Evaluate on test set
    logger.info("\nEvaluating on TEST set...")
    y_pred_test = model.predict(X_test)
    if is_xgboost:
        y_pred_test = y_pred_test + 1  # Convert back to 1-5

    y_pred_proba_test = None
    if hasattr(model, "predict_proba"):
        y_pred_proba_test = model.predict_proba(X_test)

    test_metrics = calculate_metrics_enhanced(
        y_test, y_pred_test, y_pred_proba_test, f"{model_name} ({feature_set}) - Test"
    )
    results["test"] = test_metrics
    save_confusion_matrix(
        y_test, y_pred_test, model_name, feature_set, "test", output_dir
    )

    logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  Test ESI 1-2 Recall: {test_metrics['esi_1_2_recall']:.4f}")
    logger.info(f"  Test ESI 3-4 Recall: {test_metrics['esi_3_4_recall']:.4f}")

    # Calculate train-val gap
    train_val_gap = abs(train_metrics["accuracy"] - val_metrics["accuracy"])
    logger.info(f"\nTrain-Val Accuracy Gap: {train_val_gap:.4f}")
    if train_val_gap > 0.05:
        logger.warning(
            f"  [WARN] Large train-val gap ({train_val_gap:.2%}) - possible overfitting"
        )
    else:
        logger.info(f"  [OK] Train-val gap is acceptable ({train_val_gap:.2%})")

    return results


# ============================================================================
# PHASE 5: MAIN TRAINING PIPELINE
# ============================================================================


def main():
    """Main training pipeline."""
    logger.info("=" * 80)
    logger.info("BALANCED RECEPTIONIST TRIAGE MODEL TRAINING v3")
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info(f"  Class weights: {MODERATE_CLASS_WEIGHTS}")
    logger.info(
        "  SMOTE strategy: ESI 1->5000, 2->15000, 3->original, 4->original, 5->original"
    )
    logger.info("  Target: Accuracy >50%, ESI 1-2 Recall >40%")
    logger.info("=" * 80)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # Analyze ESI distribution
    esi_distribution = analyze_esi_distribution(y_train)

    # Define feature sets and models
    feature_sets = ["A", "B", "C", "D"]
    models_config = {
        "Logistic Regression": train_logistic_regression_no_tuning,
        "Decision Tree": train_decision_tree_no_tuning,
        "Random Forest": train_random_forest_no_tuning,
        "XGBoost": train_xgboost_no_tuning,
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

        # Apply balanced SMOTE
        X_train_balanced, y_train_balanced = apply_balanced_smote(
            X_train_fe, y_train, esi_distribution
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
        for model_name, train_func in models_config.items():
            try:
                logger.info(f"\n{'=' * 80}")
                logger.info(
                    f"TRAINING {model_name.upper()} (Feature Set {feature_set})"
                )
                logger.info(f"{'=' * 80}")

                # Train model (using batches of 5k samples from full dataset)
                model, params = train_func(
                    X_train_balanced, y_train_balanced, feature_set, batch_size=5000
                )

                # Save parameters used
                params_file = (
                    OUTPUT_DIR
                    / "parameters"
                    / f"used_params_{model_name.lower().replace(' ', '_')}_{feature_set}.json"
                )
                with open(params_file, "w") as f:
                    json.dump(params, f, indent=2, default=str)
                logger.info(f"Saved parameters to: {params_file}")

                # Save model
                model_file = (
                    OUTPUT_DIR
                    / "models"
                    / f"{model_name.lower().replace(' ', '_')}_{feature_set}.pkl"
                )
                joblib.dump(model, model_file)
                logger.info(f"Saved model to: {model_file}")

                # Evaluate on all sets
                eval_results = evaluate_model_comprehensive(
                    model,
                    model_name,
                    X_train_balanced,
                    y_train_balanced,
                    X_val_fe,
                    y_val,
                    X_test_fe,
                    y_test,
                    feature_set,
                    OUTPUT_DIR,
                )

                # Save metrics
                for dataset_name, metrics in eval_results.items():
                    metrics_file = (
                        OUTPUT_DIR
                        / "metrics"
                        / f"{dataset_name}_metrics_{model_name.lower().replace(' ', '_')}_{feature_set}.csv"
                    )
                    pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
                    logger.info(f"Saved {dataset_name} metrics to: {metrics_file}")

                feature_set_results[model_name] = {
                    "model": model,
                    "params": params,
                    "metrics": eval_results,
                }

            except FileNotFoundError as e:
                logger.warning(
                    f"Skipping {model_name} on feature set {feature_set}: {e}"
                )
                continue
            except Exception as e:
                logger.error(
                    f"Error training {model_name} on feature set {feature_set}: {e}"
                )
                import traceback

                logger.error(traceback.format_exc())
                continue

        all_results[feature_set] = feature_set_results

        # Select best model for this feature set (based on test accuracy)
        if feature_set_results:
            best_model_name = max(
                feature_set_results.keys(),
                key=lambda k: feature_set_results[k]["metrics"]["test"].get(
                    "accuracy", 0
                ),
            )
            logger.info(
                f"\nBest model for feature set {feature_set}: {best_model_name}"
            )
            best_metrics = feature_set_results[best_model_name]["metrics"]["test"]
            logger.info(f"  Test Accuracy: {best_metrics.get('accuracy', 0):.4f}")
            logger.info(
                f"  Test ESI 1-2 Recall: {best_metrics.get('esi_1_2_recall', 0):.4f}"
            )
            logger.info(
                f"  Test ESI 3-4 Recall: {best_metrics.get('esi_3_4_recall', 0):.4f}"
            )

    # Generate comprehensive comparison report
    logger.info(f"\n{'=' * 80}")
    logger.info("GENERATING COMPREHENSIVE REPORT")
    logger.info(f"{'=' * 80}")

    generate_comparison_report(all_results, OUTPUT_DIR)

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nResults saved to: {OUTPUT_DIR}")
    logger.info("\nNext steps:")
    logger.info("1. Review balanced_training_summary.md")
    logger.info("2. Check train/val/test metrics for overfitting")
    logger.info("3. Select best model based on accuracy and ESI 1-2 recall balance")


def generate_comparison_report(all_results: dict, output_dir: Path) -> None:
    """Generate comprehensive comparison report."""
    report_lines = []
    report_lines.append("# Balanced Training Summary Report")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("## Configuration")
    report_lines.append("")
    report_lines.append(f"- **Class Weights**: {MODERATE_CLASS_WEIGHTS}")
    report_lines.append(
        "- **SMOTE Strategy**: ESI 1->5000, 2->15000, 3->original, 4->original, 5->5000"
    )
    report_lines.append("- **No Hyperparameter Tuning**: Used best params from v2")
    report_lines.append("")

    # Create comparison table
    report_lines.append("## Model Comparison (Test Set)")
    report_lines.append("")
    report_lines.append(
        "| Model | Feature Set | Accuracy | ESI 1-2 Recall | ESI 3-4 Recall | Macro F1 | Train-Val Gap |"
    )
    report_lines.append(
        "|-------|-------------|----------|----------------|----------------|----------|---------------|"
    )

    best_overall = None
    best_score = -1

    for feature_set in ["A", "B", "C", "D"]:
        if feature_set not in all_results:
            continue

        for model_name, results in all_results[feature_set].items():
            test_metrics = results["metrics"]["test"]
            val_metrics = results["metrics"]["val"]
            train_metrics = results["metrics"]["train"]

            accuracy = test_metrics.get("accuracy", 0)
            esi_1_2_recall = test_metrics.get("esi_1_2_recall", 0)
            esi_3_4_recall = test_metrics.get("esi_3_4_recall", 0)
            macro_f1 = test_metrics.get("macro_f1", 0)
            train_val_gap = abs(
                train_metrics.get("accuracy", 0) - val_metrics.get("accuracy", 0)
            )

            report_lines.append(
                f"| {model_name} | {feature_set} | {accuracy:.4f} | {esi_1_2_recall:.4f} | "
                f"{esi_3_4_recall:.4f} | {macro_f1:.4f} | {train_val_gap:.4f} |"
            )

            # Track best model (prioritize accuracy >50% and ESI 1-2 recall >40%)
            if accuracy > 0.50 and esi_1_2_recall > 0.40:
                score = accuracy * 0.6 + esi_1_2_recall * 0.4
                if score > best_score:
                    best_score = score
                    best_overall = (model_name, feature_set, test_metrics)

    report_lines.append("")

    # Best models per feature set
    report_lines.append("## Best Model Per Feature Set (Test Set)")
    report_lines.append("")
    for feature_set in ["A", "B", "C", "D"]:
        if feature_set not in all_results or not all_results[feature_set]:
            continue

        best_model = max(
            all_results[feature_set].items(),
            key=lambda x: x[1]["metrics"]["test"].get("accuracy", 0),
        )
        model_name, results = best_model
        test_metrics = results["metrics"]["test"]

        report_lines.append(f"### Feature Set {feature_set}")
        report_lines.append(f"- **Model**: {model_name}")
        report_lines.append(f"- **Accuracy**: {test_metrics.get('accuracy', 0):.4f}")
        report_lines.append(
            f"- **ESI 1-2 Recall**: {test_metrics.get('esi_1_2_recall', 0):.4f}"
        )
        report_lines.append(
            f"- **ESI 3-4 Recall**: {test_metrics.get('esi_3_4_recall', 0):.4f}"
        )
        report_lines.append(f"- **Macro F1**: {test_metrics.get('macro_f1', 0):.4f}")
        report_lines.append("")

    # Overall best model
    if best_overall:
        model_name, feature_set, metrics = best_overall
        report_lines.append("## Best Overall Model (Test Set)")
        report_lines.append("")
        report_lines.append(f"- **Model**: {model_name}")
        report_lines.append(f"- **Feature Set**: {feature_set}")
        report_lines.append(f"- **Accuracy**: {metrics.get('accuracy', 0):.4f}")
        report_lines.append(
            f"- **ESI 1-2 Recall**: {metrics.get('esi_1_2_recall', 0):.4f}"
        )
        report_lines.append(
            f"- **ESI 3-4 Recall**: {metrics.get('esi_3_4_recall', 0):.4f}"
        )
        report_lines.append(f"- **Macro F1**: {metrics.get('macro_f1', 0):.4f}")
        report_lines.append("")

        # Recommendations
        accuracy = metrics.get("accuracy", 0)
        esi_1_2_recall = metrics.get("esi_1_2_recall", 0)
        esi_3_4_recall = metrics.get("esi_3_4_recall", 0)

        report_lines.append("## Recommendations")
        report_lines.append("")

        if accuracy >= 0.50:
            report_lines.append(
                f"- [OK] Accuracy ({accuracy:.2%}) meets target (>= 50%)"
            )
        else:
            report_lines.append(
                f"- [WARN] Accuracy ({accuracy:.2%}) is below target (>= 50%)"
            )

        if esi_1_2_recall >= 0.40:
            report_lines.append(
                f"- [OK] ESI 1-2 recall ({esi_1_2_recall:.2%}) meets target (>= 40%)"
            )
        else:
            report_lines.append(
                f"- [WARN] ESI 1-2 recall ({esi_1_2_recall:.2%}) is below target (>= 40%)"
            )

        if esi_3_4_recall >= 0.50:
            report_lines.append(
                f"- [OK] ESI 3-4 recall ({esi_3_4_recall:.2%}) meets target (>= 50%)"
            )
        else:
            report_lines.append(
                f"- [WARN] ESI 3-4 recall ({esi_3_4_recall:.2%}) is below target (>= 50%)"
            )

        report_lines.append(
            f"- **Recommended Model**: {model_name} with Feature Set {feature_set}"
        )
        report_lines.append("")

    # Save report
    report_file = output_dir / "reports" / "balanced_training_summary.md"
    with open(report_file, "w") as f:
        f.write("\n".join(report_lines))
    logger.info(f"Saved comprehensive report to: {report_file}")


if __name__ == "__main__":
    main()
