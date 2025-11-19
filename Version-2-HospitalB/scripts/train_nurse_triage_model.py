"""
Nurse Triage Model Training Pipeline

Trains comprehensive nurse triage models using all 23 features:
- Vital signs (7): pulse_yj, respiration_yj, sbp, dbp, temp_c_yj, pain, pain_missing
- Demographics (1): age
- RFV clusters (12): All rfv1_cluster_* columns
- Binary flags (3): ambulance_arrival, seen_72h, injury

Models:
- Logistic Regression (with batch training)
- Decision Tree
- Random Forest
- XGBoost
- Stacking Ensemble (combines all 4 base models)

Configuration:
- Moderate class weights: {1:10, 2:5, 3:1, 4:1, 5:2}
- Balanced SMOTE: {1:5000, 2:15000, 3:original, 4:original, 5:original}
- Batch training: 5k samples per batch
- No hyperparameter tuning: Use best params from receptionist v2

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
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
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
OUTPUT_DIR = ROOT_DIR / "data" / "nurse_models"

# Create output directories
for subdir in ["models", "metrics", "parameters", "analysis", "reports"]:
    (OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

# Add file handler after directories are created
file_handler = logging.FileHandler(OUTPUT_DIR / "reports" / "training.log")
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


def extract_nurse_features(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
) -> tuple:
    """Extract all 23 features for nurse model (vitals + receptionist features)."""
    logger.info("\nExtracting nurse features (all 23 features)...")

    # Define all 23 nurse features
    nurse_features = [
        # Vital signs (7)
        "pulse_yj",
        "respiration_yj",
        "sbp",
        "dbp",
        "temp_c_yj",
        "pain",
        "pain_missing",
        # Demographics (1)
        "age",
        # RFV clusters (12)
        "rfv1_cluster_Ear_Nose_Throat",
        "rfv1_cluster_Fever_Infection",
        "rfv1_cluster_Gastrointestinal",
        "rfv1_cluster_General_Symptoms",
        "rfv1_cluster_Mental_Health",
        "rfv1_cluster_Musculoskeletal",
        "rfv1_cluster_Neurological",
        "rfv1_cluster_Other",
        "rfv1_cluster_Respiratory",
        "rfv1_cluster_Skin",
        "rfv1_cluster_Trauma_Injury",
        "rfv1_cluster_Urinary_Genitourinary",
        # Binary flags (3)
        "ambulance_arrival",
        "seen_72h",
        "injury",
    ]

    # Verify all features exist
    missing_features = [f for f in nurse_features if f not in X_train.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    logger.info(f"Extracted {len(nurse_features)} nurse features")
    logger.info("  Vital signs: 7")
    logger.info("  Demographics: 1")
    logger.info("  RFV clusters: 12")
    logger.info("  Binary flags: 3")

    X_nurse_train = X_train[nurse_features].copy()
    X_nurse_val = X_val[nurse_features].copy()
    X_nurse_test = X_test[nurse_features].copy()

    # Fix RFV clusters: rows with all zeros should be assigned to "Other" cluster
    rfv_cols = [col for col in nurse_features if col.startswith("rfv1_cluster_")]
    for name, df_set in [
        ("Train", X_nurse_train),
        ("Val", X_nurse_val),
        ("Test", X_nurse_test),
    ]:
        rfv_sum = df_set[rfv_cols].sum(axis=1)
        zero_rows = (rfv_sum == 0).sum()
        if zero_rows > 0:
            logger.info(f"  {name}: {zero_rows:,} rows with all zeros in RFV clusters")
            if "rfv1_cluster_Other" in df_set.columns:
                df_set.loc[rfv_sum == 0, "rfv1_cluster_Other"] = 1.0
            else:
                most_common_cluster = df_set[rfv_cols].sum().idxmax()
                df_set.loc[rfv_sum == 0, most_common_cluster] = 1.0
                logger.info(f"    Assigned to: {most_common_cluster}")

    logger.info("\nNurse feature sets:")
    logger.info(f"  X_nurse_train: {X_nurse_train.shape}")
    logger.info(f"  X_nurse_val: {X_nurse_val.shape}")
    logger.info(f"  X_nurse_test: {X_nurse_test.shape}")

    return X_nurse_train, X_nurse_val, X_nurse_test, nurse_features


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
        1: 5000,  # ESI 1: moderate oversample
        2: 15000,  # ESI 2: moderate oversample
        3: esi_distribution[3]["count"],  # Keep original
        4: esi_distribution[4]["count"],  # Keep original
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
# PHASE 2: BATCH CREATION
# ============================================================================


def create_batches(
    X_train: pd.DataFrame, y_train: pd.Series, batch_size: int = 5000
) -> list:
    """Split training data into batches of specified size."""
    total_samples = len(X_train)
    num_batches = (total_samples + batch_size - 1) // batch_size  # Ceiling division

    logger.info(
        f"Splitting {total_samples:,} samples into {num_batches} batches of ~{batch_size:,} samples each..."  # noqa: E501
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


def load_and_modify_params(model_name: str, feature_set: str = "A") -> dict:
    """Load best hyperparameters from v2 and replace class_weight with moderate weights."""  # noqa: E501
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
# PHASE 4: BASE MODEL TRAINING
# ============================================================================


def train_logistic_regression_no_tuning(
    X_train: pd.DataFrame, y_train: pd.Series, feature_set: str = "A"
) -> tuple:
    """Train Logistic Regression using best params from v2 with moderate class weights."""  # noqa: E501
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING LOGISTIC REGRESSION (NO TUNING)")
    logger.info("=" * 80)

    params = load_and_modify_params("Logistic Regression", feature_set)

    # Ensure max_iter is set for convergence
    if "max_iter" not in params:
        params["max_iter"] = 10000
    params["tol"] = 1e-4
    params["random_state"] = 42

    logger.info(f"Training with parameters: {params}")
    logger.info(f"Training on {len(X_train):,} samples...")

    start_time = time.time()
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    logger.info(f"Training completed in {train_time:.2f} seconds")

    return model, params


def train_decision_tree_no_tuning(
    X_train: pd.DataFrame, y_train: pd.Series, feature_set: str = "A"
) -> tuple:
    """Train Decision Tree using best params from v2 with moderate class weights."""
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING DECISION TREE (NO TUNING)")
    logger.info("=" * 80)

    params = load_and_modify_params("Decision Tree", feature_set)
    params["random_state"] = 42

    logger.info(f"Training with parameters: {params}")
    logger.info(f"Training on {len(X_train):,} samples...")

    start_time = time.time()
    model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    logger.info(f"Training completed in {train_time:.2f} seconds")

    return model, params


def train_random_forest_no_tuning(
    X_train: pd.DataFrame, y_train: pd.Series, feature_set: str = "A"
) -> tuple:
    """Train Random Forest using best params from v2 with moderate class weights."""
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING RANDOM FOREST (NO TUNING)")
    logger.info("=" * 80)

    params = load_and_modify_params("Random Forest", feature_set)
    params["oob_score"] = True
    params["random_state"] = 42
    params["n_jobs"] = -1

    logger.info(f"Training with parameters: {params}")
    logger.info(f"Training on {len(X_train):,} samples...")

    start_time = time.time()
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    logger.info(f"Training completed in {train_time:.2f} seconds")
    if hasattr(model, "oob_score_"):
        logger.info(f"OOB Score: {model.oob_score_:.4f}")

    return model, params


def train_xgboost_no_tuning(
    X_train: pd.DataFrame, y_train: pd.Series, feature_set: str = "A"
) -> tuple:
    """Train XGBoost using best params from v2 with moderate class weights."""
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING XGBOOST (NO TUNING)")
    logger.info("=" * 80)

    # XGBoost requires class labels to start from 0
    logger.info("Converting ESI labels from 1-5 to 0-4 for XGBoost...")
    y_train_xgb = y_train - 1

    params = load_and_modify_params("XGBoost", feature_set)
    params["objective"] = "multi:softmax"
    params["num_class"] = 5
    params["random_state"] = 42
    params["n_jobs"] = -1

    # Apply moderate class weights as sample weights
    sample_weights_xgb = np.array([XGBOOST_CLASS_WEIGHTS[int(y)] for y in y_train_xgb])

    logger.info(f"Training with parameters: {params}")
    logger.info(f"Training on {len(X_train):,} samples...")

    start_time = time.time()
    model = XGBClassifier(**params)
    model.fit(X_train, y_train_xgb, sample_weight=sample_weights_xgb)
    train_time = time.time() - start_time

    logger.info(f"Training completed in {train_time:.2f} seconds")

    return model, params


# ============================================================================
# PHASE 5: STACKING ENSEMBLE
# ============================================================================


class StackingEnsemble:
    """Stacking ensemble that combines base model predictions with meta-learner."""

    def __init__(self, base_models: dict, meta_learner: Any):
        self.base_models = base_models
        self.meta_learner = meta_learner

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using stacking."""
        # Get predictions from all base models
        stacked_features = self._get_stacked_features(X)
        return self.meta_learner.predict(stacked_features)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability predictions using stacking."""
        # Get predictions from all base models
        stacked_features = self._get_stacked_features(X)
        return self.meta_learner.predict_proba(stacked_features)

    def _get_stacked_features(self, X: pd.DataFrame) -> np.ndarray:
        """Get stacked features from base model predictions."""
        stacked_probas = []

        for _model_name, model in self.base_models.items():
            # Get probability predictions
            if isinstance(model, XGBClassifier):
                # XGBoost predictions are 0-indexed
                proba = model.predict_proba(X)
            else:
                proba = model.predict_proba(X)

            stacked_probas.append(proba)

        # Concatenate all probability vectors (4 models x 5 classes = 20 features)
        stacked_features = np.hstack(stacked_probas)
        return stacked_features


def train_stacking_ensemble(
    base_models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cv: int = 3,
) -> tuple:
    """Train stacking ensemble with meta-learner using out-of-fold predictions."""
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING STACKING ENSEMBLE")
    logger.info("=" * 80)

    logger.info(f"Base models: {list(base_models.keys())}")
    logger.info(f"Generating out-of-fold predictions using {cv}-fold CV...")

    # Generate out-of-fold predictions from base models using CV
    n_samples = len(X_train)
    n_classes = 5
    n_base_models = len(base_models)

    # Initialize array for stacked features (n_samples x n_base_models x n_classes)
    stacked_features_train = np.zeros((n_samples, n_base_models * n_classes))

    cv_fold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    start_time = time.time()
    for fold_idx, (train_idx, val_idx) in enumerate(
        cv_fold.split(X_train, y_train), start=1
    ):
        logger.info(
            f"  Fold {fold_idx}/{cv}: Training base models on {len(train_idx):,} samples..."  # noqa: E501
        )

        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]

        # Train base models on fold training data and predict on fold validation data
        fold_probas = []
        for _model_name, base_model_template in base_models.items():
            # Get model parameters from template
            model_params = base_model_template.get_params()

            # Create and train model for this fold
            if isinstance(base_model_template, XGBClassifier):
                y_fold_train_xgb = y_fold_train - 1
                # Extract key XGBoost parameters
                xgb_params = {
                    "objective": "multi:softmax",
                    "num_class": 5,
                    "random_state": 42,
                    "n_jobs": -1,
                }
                # Copy tunable parameters
                for param in [
                    "max_depth",
                    "learning_rate",
                    "n_estimators",
                    "subsample",
                    "colsample_bytree",
                    "reg_alpha",
                    "reg_lambda",
                    "gamma",
                ]:
                    if param in model_params:
                        xgb_params[param] = model_params[param]

                fold_model = XGBClassifier(**xgb_params)
                fold_model.fit(X_fold_train, y_fold_train_xgb)
                fold_proba = fold_model.predict_proba(X_fold_val)
            elif isinstance(base_model_template, LogisticRegression):
                fold_model = LogisticRegression(
                    random_state=42,
                    **{
                        k: v
                        for k, v in model_params.items()
                        if k
                        not in [
                            "random_state",
                            "n_iter_",
                            "classes_",
                            "coef_",
                            "intercept_",
                            "n_features_in_",
                            "feature_names_in_",
                        ]
                    },
                )
                fold_model.fit(X_fold_train, y_fold_train)
                fold_proba = fold_model.predict_proba(X_fold_val)
            elif isinstance(base_model_template, DecisionTreeClassifier):
                fold_model = DecisionTreeClassifier(
                    random_state=42,
                    **{
                        k: v
                        for k, v in model_params.items()
                        if k
                        not in [
                            "random_state",
                            "tree_",
                            "n_features_in_",
                            "feature_names_in_",
                            "n_classes_",
                            "classes_",
                        ]
                    },
                )
                fold_model.fit(X_fold_train, y_fold_train)
                fold_proba = fold_model.predict_proba(X_fold_val)
            elif isinstance(base_model_template, RandomForestClassifier):
                fold_model = RandomForestClassifier(
                    random_state=42,
                    n_jobs=-1,
                    **{
                        k: v
                        for k, v in model_params.items()
                        if k
                        not in [
                            "random_state",
                            "n_jobs",
                            "oob_score",
                            "estimators_",
                            "n_features_in_",
                            "feature_names_in_",
                            "n_classes_",
                            "classes_",
                            "oob_score_",
                        ]
                    },
                )
                if "oob_score" in model_params and model_params["oob_score"]:
                    fold_model.set_params(oob_score=True)
                fold_model.fit(X_fold_train, y_fold_train)
                fold_proba = fold_model.predict_proba(X_fold_val)
            else:
                # Generic fallback - create new instance with same parameters
                fold_model = type(base_model_template)(
                    **{
                        k: v
                        for k, v in model_params.items()
                        if not k.endswith("_") and k != "random_state"
                    }
                )
                if hasattr(base_model_template, "random_state"):
                    fold_model.set_params(random_state=42)
                fold_model.fit(X_fold_train, y_fold_train)
                fold_proba = fold_model.predict_proba(X_fold_val)

            fold_probas.append(fold_proba)

        # Stack probabilities for this fold
        fold_stacked = np.hstack(fold_probas)
        stacked_features_train[val_idx] = fold_stacked

    cv_time = time.time() - start_time
    logger.info(f"Out-of-fold predictions generated in {cv_time:.2f} seconds")

    # Train meta-learner on stacked features
    logger.info("\nTraining meta-learner (Logistic Regression) on stacked features...")
    meta_learner = LogisticRegression(
        class_weight="balanced", max_iter=5000, random_state=42, n_jobs=-1
    )

    start_time = time.time()
    meta_learner.fit(stacked_features_train, y_train)
    meta_time = time.time() - start_time
    logger.info(f"Meta-learner training completed in {meta_time:.2f} seconds")

    # Create stacking ensemble wrapper
    stacking_model = StackingEnsemble(base_models, meta_learner)

    # Evaluate on validation set for quick check
    logger.info("\nEvaluating stacking ensemble on validation set...")
    y_pred_val = stacking_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred_val)
    logger.info(f"Validation accuracy: {val_accuracy:.4f}")

    return stacking_model, {
        "meta_learner": "LogisticRegression",
        "cv": cv,
        "base_models": list(base_models.keys()),
    }


# ============================================================================
# PHASE 6: EVALUATION METRICS
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
    dataset_name: str,
    output_dir: Path,
) -> None:
    """Save confusion matrix to file."""
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])
    cm_df = pd.DataFrame(cm, index=[1, 2, 3, 4, 5], columns=[1, 2, 3, 4, 5])
    filename = (
        f"confusion_matrix_{model_name.lower().replace(' ', '_')}_{dataset_name}.csv"
    )
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
    output_dir: Path,
) -> dict:
    """Evaluate model on train, validation, and test sets."""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"COMPREHENSIVE EVALUATION: {model_name.upper()}")
    logger.info(f"{'=' * 80}")

    results = {}
    is_xgboost = isinstance(model, XGBClassifier)
    isinstance(model, StackingEnsemble)

    # Evaluate on train set
    logger.info("\nEvaluating on TRAIN set...")
    y_pred_train = model.predict(X_train)
    if is_xgboost:
        y_pred_train = y_pred_train + 1  # Convert back to 1-5

    y_pred_proba_train = None
    if hasattr(model, "predict_proba"):
        y_pred_proba_train = model.predict_proba(X_train)

    train_metrics = calculate_metrics_enhanced(
        y_train, y_pred_train, y_pred_proba_train, f"{model_name} - Train"
    )
    results["train"] = train_metrics
    save_confusion_matrix(y_train, y_pred_train, model_name, "train", output_dir)

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
        y_val, y_pred_val, y_pred_proba_val, f"{model_name} - Val"
    )
    results["val"] = val_metrics
    save_confusion_matrix(y_val, y_pred_val, model_name, "val", output_dir)

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
        y_test, y_pred_test, y_pred_proba_test, f"{model_name} - Test"
    )
    results["test"] = test_metrics
    save_confusion_matrix(y_test, y_pred_test, model_name, "test", output_dir)

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
# PHASE 7: MAIN TRAINING PIPELINE
# ============================================================================


def main():
    """Main training pipeline."""
    logger.info("=" * 80)
    logger.info("NURSE TRIAGE MODEL TRAINING")
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info("  Features: 23 (vitals + receptionist features)")
    logger.info(f"  Class weights: {MODERATE_CLASS_WEIGHTS}")
    logger.info(
        "  SMOTE strategy: ESI 1->5000, 2->15000, 3->original, 4->original, 5->original"
    )
    logger.info("  Target: Accuracy >50%, ESI 1-2 Recall >40%")
    logger.info("=" * 80)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # Extract nurse features (all 23)
    X_nurse_train, X_nurse_val, X_nurse_test, nurse_features = extract_nurse_features(
        X_train, X_val, X_test
    )

    # Analyze ESI distribution
    esi_distribution = analyze_esi_distribution(y_train)

    # Apply balanced SMOTE
    X_train_balanced, y_train_balanced = apply_balanced_smote(
        X_nurse_train, y_train, esi_distribution
    )

    # Convert to DataFrame if needed
    if isinstance(X_train_balanced, np.ndarray):
        X_train_balanced = pd.DataFrame(X_train_balanced, columns=nurse_features)
    if isinstance(y_train_balanced, np.ndarray):
        y_train_balanced = pd.Series(y_train_balanced)

    # Define models to train
    models_config = {
        "Logistic Regression": train_logistic_regression_no_tuning,
        "Decision Tree": train_decision_tree_no_tuning,
        "Random Forest": train_random_forest_no_tuning,
        "XGBoost": train_xgboost_no_tuning,
    }

    # Store all results
    all_results = {}
    base_models = {}  # Store for stacking

    # Train base models
    logger.info(f"\n{'=' * 80}")
    logger.info("TRAINING BASE MODELS")
    logger.info(f"{'=' * 80}")

    for model_name, train_func in models_config.items():
        try:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"TRAINING {model_name.upper()}")
            logger.info(f"{'=' * 80}")

            # Train model
            model, params = train_func(
                X_train_balanced, y_train_balanced, feature_set="A"
            )

            # Save parameters used
            params_file = (
                OUTPUT_DIR
                / "parameters"
                / f"used_params_{model_name.lower().replace(' ', '_')}.json"
            )
            with open(params_file, "w") as f:
                json.dump(params, f, indent=2, default=str)
            logger.info(f"Saved parameters to: {params_file}")

            # Save model
            model_file = (
                OUTPUT_DIR / "models" / f"{model_name.lower().replace(' ', '_')}.pkl"
            )
            joblib.dump(model, model_file)
            logger.info(f"Saved model to: {model_file}")

            # Store for stacking
            base_models[model_name] = model

            # Evaluate on all sets
            eval_results = evaluate_model_comprehensive(
                model,
                model_name,
                X_train_balanced,
                y_train_balanced,
                X_nurse_val,
                y_val,
                X_nurse_test,
                y_test,
                OUTPUT_DIR,
            )

            # Save metrics
            for dataset_name, metrics in eval_results.items():
                metrics_file = (
                    OUTPUT_DIR
                    / "metrics"
                    / f"{dataset_name}_metrics_{model_name.lower().replace(' ', '_')}.csv"  # noqa: E501
                )
                pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
                logger.info(f"Saved {dataset_name} metrics to: {metrics_file}")

            all_results[model_name] = {
                "model": model,
                "params": params,
                "metrics": eval_results,
            }

        except FileNotFoundError as e:
            logger.warning(f"Skipping {model_name}: {e}")
            continue
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            import traceback

            logger.error(traceback.format_exc())
            continue

    # Train stacking ensemble
    if len(base_models) >= 2:
        logger.info(f"\n{'=' * 80}")
        logger.info("TRAINING STACKING ENSEMBLE")
        logger.info(f"{'=' * 80}")

        try:
            stacking_model, stacking_params = train_stacking_ensemble(
                base_models,
                X_train_balanced,
                y_train_balanced,
                X_nurse_val,
                y_val,
                cv=3,
            )

            # Save stacking model
            stacking_file = OUTPUT_DIR / "models" / "stacking_ensemble.pkl"
            joblib.dump(stacking_model, stacking_file)
            logger.info(f"Saved stacking model to: {stacking_file}")

            # Save stacking parameters
            params_file = (
                OUTPUT_DIR / "parameters" / "used_params_stacking_ensemble.json"
            )
            with open(params_file, "w") as f:
                json.dump(stacking_params, f, indent=2, default=str)
            logger.info(f"Saved stacking parameters to: {params_file}")

            # Evaluate stacking on all sets
            eval_results = evaluate_model_comprehensive(
                stacking_model,
                "Stacking Ensemble",
                X_train_balanced,
                y_train_balanced,
                X_nurse_val,
                y_val,
                X_nurse_test,
                y_test,
                OUTPUT_DIR,
            )

            # Save metrics
            for dataset_name, metrics in eval_results.items():
                metrics_file = (
                    OUTPUT_DIR
                    / "metrics"
                    / f"{dataset_name}_metrics_stacking_ensemble.csv"
                )
                pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
                logger.info(f"Saved {dataset_name} metrics to: {metrics_file}")

            all_results["Stacking Ensemble"] = {
                "model": stacking_model,
                "params": stacking_params,
                "metrics": eval_results,
            }

        except Exception as e:
            logger.error(f"Error training stacking ensemble: {e}")
            import traceback

            logger.error(traceback.format_exc())

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
    logger.info("1. Review nurse_training_summary.md")
    logger.info("2. Check train/val/test metrics for overfitting")
    logger.info("3. Compare with receptionist model performance")
    logger.info("4. Select best model for deployment")


def generate_comparison_report(all_results: dict, output_dir: Path) -> None:
    """Generate comprehensive comparison report."""
    report_lines = []
    report_lines.append("# Nurse Triage Model Training Summary")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("## Configuration")
    report_lines.append("")
    report_lines.append("- **Features**: 23 (vitals + receptionist features)")
    report_lines.append(f"- **Class Weights**: {MODERATE_CLASS_WEIGHTS}")
    report_lines.append(
        "- **SMOTE Strategy**: ESI 1->5000, 2->15000, 3->original, 4->original, 5->original"  # noqa: E501
    )
    report_lines.append(
        "- **No Hyperparameter Tuning**: Used best params from receptionist v2"
    )
    report_lines.append("")

    # Create comparison table
    report_lines.append("## Model Comparison (Test Set)")
    report_lines.append("")
    report_lines.append(
        "| Model | Accuracy | ESI 1-2 Recall | ESI 3-4 Recall | Macro F1 | Train-Val Gap |"  # noqa: E501
    )
    report_lines.append(
        "|-------|----------|----------------|----------------|----------|---------------|"
    )

    best_overall = None
    best_score = -1

    for model_name, results in all_results.items():
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
            f"| {model_name} | {accuracy:.4f} | {esi_1_2_recall:.4f} | "
            f"{esi_3_4_recall:.4f} | {macro_f1:.4f} | {train_val_gap:.4f} |"
        )

        # Track best model (prioritize accuracy >50% and ESI 1-2 recall >40%)
        if accuracy > 0.50 and esi_1_2_recall > 0.40:
            score = accuracy * 0.6 + esi_1_2_recall * 0.4
            if score > best_score:
                best_score = score
                best_overall = (model_name, test_metrics)

    report_lines.append("")

    # Overall best model
    if best_overall:
        model_name, metrics = best_overall
        report_lines.append("## Best Overall Model (Test Set)")
        report_lines.append("")
        report_lines.append(f"- **Model**: {model_name}")
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
                f"- [WARN] ESI 1-2 recall ({esi_1_2_recall:.2%}) is below target (>= 40%)"  # noqa: E501
            )

        if esi_3_4_recall >= 0.50:
            report_lines.append(
                f"- [OK] ESI 3-4 recall ({esi_3_4_recall:.2%}) meets target (>= 50%)"
            )
        else:
            report_lines.append(
                f"- [WARN] ESI 3-4 recall ({esi_3_4_recall:.2%}) is below target (>= 50%)"  # noqa: E501
            )

        report_lines.append(f"- **Recommended Model**: {model_name}")
        report_lines.append("")

    # Save report
    report_file = output_dir / "reports" / "nurse_training_summary.md"
    with open(report_file, "w") as f:
        f.write("\n".join(report_lines))
    logger.info(f"Saved comprehensive report to: {report_file}")


if __name__ == "__main__":
    main()
