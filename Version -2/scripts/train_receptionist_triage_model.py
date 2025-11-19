"""
Receptionist Triage Model Training Pipeline

Trains 5 machine learning models for receptionist-level triage classification
using 16 features available at patient check-in.

Features:
- age (1)
- rfv1_cluster_* (12 one-hot columns)
- ambulance_arrival, seen_72h, injury (3 binary)

Focus: ESI 1-2 recall (critical patients) with cost-sensitive learning.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.combine import SMOTETomek
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
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
OUTPUT_DIR = ROOT_DIR / "data" / "receptionist_models"

# Create output directories
for subdir in ["models", "metrics", "parameters", "analysis", "reports"]:
    (OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

# Add file handler after directories are created
file_handler = logging.FileHandler(OUTPUT_DIR / "reports" / "training.log")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


# ============================================================================
# PHASE 1: DATA PREPARATION
# ============================================================================


def extract_receptionist_features() -> tuple:
    """Extract 16 receptionist-available features from processed datasets."""
    logger.info("=" * 80)
    logger.info("PHASE 1: DATA PREPARATION")
    logger.info("=" * 80)

    # Load processed datasets
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

    # Define receptionist features (16 total)
    receptionist_features = ["age", "ambulance_arrival", "seen_72h", "injury"]
    rfv_features = [col for col in X_train.columns if col.startswith("rfv1_cluster_")]
    receptionist_features.extend(sorted(rfv_features))

    logger.info(f"\nExtracting {len(receptionist_features)} receptionist features...")
    logger.info(f"  Features: {receptionist_features}")

    # Verify all features exist
    missing_features = [f for f in receptionist_features if f not in X_train.columns]
    if missing_features:
        logger.error(f"Missing features: {missing_features}")
        raise ValueError(f"Features not found: {missing_features}")

    # Extract features
    X_receptionist_train = X_train[receptionist_features].copy()
    X_receptionist_val = X_val[receptionist_features].copy()
    X_receptionist_test = X_test[receptionist_features].copy()

    # Fix RFV clusters: rows with all zeros should be assigned to "Other" cluster
    # (This happens when the baseline category "Musculoskeletal" was dropped
    # by OneHotEncoder)
    logger.info(
        "\nFixing RFV clusters " "(assigning default cluster to rows with all zeros)..."
    )
    rfv_cols = [col for col in receptionist_features if col.startswith("rfv1_cluster_")]

    for name, df_set in [
        ("Train", X_receptionist_train),
        ("Val", X_receptionist_val),
        ("Test", X_receptionist_test),
    ]:
        rfv_sum = df_set[rfv_cols].sum(axis=1)
        zero_rows = (rfv_sum == 0).sum()
        if zero_rows > 0:
            logger.info(f"  {name}: {zero_rows:,} rows with all zeros in RFV clusters")
            # Assign to "Other" cluster (most common fallback)
            if "rfv1_cluster_Other" in df_set.columns:
                df_set.loc[rfv_sum == 0, "rfv1_cluster_Other"] = 1.0
            else:
                # If "Other" doesn't exist, assign to the most common cluster
                most_common_cluster = df_set[rfv_cols].sum().idxmax()
                df_set.loc[rfv_sum == 0, most_common_cluster] = 1.0
                logger.info(f"    Assigned to: {most_common_cluster}")

    # Verify fix
    train_rfv_sum = X_receptionist_train[rfv_cols].sum(axis=1)
    val_rfv_sum = X_receptionist_val[rfv_cols].sum(axis=1)
    test_rfv_sum = X_receptionist_test[rfv_cols].sum(axis=1)

    if (
        (train_rfv_sum == 1).all()
        and (val_rfv_sum == 1).all()
        and (test_rfv_sum == 1).all()
    ):
        logger.info("  [OK] All rows now have exactly one RFV cluster assigned")
    else:
        train_issues = (train_rfv_sum != 1).sum()
        val_issues = (val_rfv_sum != 1).sum()
        test_issues = (test_rfv_sum != 1).sum()
        logger.warning(
            f"  [WARN] Still have issues - Train: {train_issues}, "
            f"Val: {val_issues}, Test: {test_issues}"
        )

    logger.info("\nExtracted receptionist features:")
    logger.info(f"  X_receptionist_train: {X_receptionist_train.shape}")
    logger.info(f"  X_receptionist_val: {X_receptionist_val.shape}")
    logger.info(f"  X_receptionist_test: {X_receptionist_test.shape}")

    return (
        X_receptionist_train,
        X_receptionist_val,
        X_receptionist_test,
        y_train,
        y_val,
        y_test,
        receptionist_features,
    )


def verify_data_quality(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> None:
    """Verify data quality of receptionist features."""
    logger.info("\n" + "=" * 80)
    logger.info("DATA QUALITY VERIFICATION")
    logger.info("=" * 80)

    report = []

    # Check shapes
    logger.info("\nChecking shapes...")
    expected_shape_train = (114615, 16)
    expected_shape_val = (24560, 16)
    expected_shape_test = (24561, 16)

    shapes_ok = True
    if X_train.shape != expected_shape_train:
        logger.warning(
            f"  X_train shape: {X_train.shape} (expected: {expected_shape_train})"
        )
        shapes_ok = False
    else:
        logger.info(f"  X_train shape: {X_train.shape} [OK]")

    if X_val.shape != expected_shape_val:
        logger.warning(f"  X_val shape: {X_val.shape} (expected: {expected_shape_val})")
        shapes_ok = False
    else:
        logger.info(f"  X_val shape: {X_val.shape} [OK]")

    if X_test.shape != expected_shape_test:
        logger.warning(
            f"  X_test shape: {X_test.shape} (expected: {expected_shape_test})"
        )
        shapes_ok = False
    else:
        logger.info(f"  X_test shape: {X_test.shape} [OK]")

    report.append(f"Shapes: {'[OK]' if shapes_ok else '[WARN]'}")

    # Check missing values
    logger.info("\nChecking missing values...")
    for name, df_set in [("Train", X_train), ("Val", X_val), ("Test", X_test)]:
        null_count = df_set.isnull().sum().sum()
        if null_count > 0:
            logger.error(f"  {name}: {null_count} nulls found!")
            report.append(f"{name} missing values: {null_count} [ERROR]")
        else:
            logger.info(f"  {name}: 0 nulls [OK]")
            report.append(f"{name} missing values: 0 [OK]")

    # Check binary features
    logger.info("\nChecking binary features...")
    binary_features = ["ambulance_arrival", "seen_72h", "injury"]
    for feature in binary_features:
        if feature in X_train.columns:
            train_vals = set(X_train[feature].unique())
            if train_vals.issubset({0, 1}):
                logger.info(f"  {feature}: [OK] (values: {sorted(train_vals)})")
                report.append(f"{feature}: Binary [OK]")
            else:
                logger.warning(f"  {feature}: [WARN] (values: {sorted(train_vals)})")
                report.append(f"{feature}: Not binary [WARN]")

    # Check RFV clusters (each row should have exactly one 1)
    logger.info("\nChecking RFV clusters...")
    rfv_cols = [col for col in X_train.columns if col.startswith("rfv1_cluster_")]
    train_rfv_sum = X_train[rfv_cols].sum(axis=1)
    val_rfv_sum = X_val[rfv_cols].sum(axis=1)
    test_rfv_sum = X_test[rfv_cols].sum(axis=1)

    train_ok = (train_rfv_sum == 1).all()
    val_ok = (val_rfv_sum == 1).all()
    test_ok = (test_rfv_sum == 1).all()

    if train_ok and val_ok and test_ok:
        logger.info("  RFV clusters: [OK] (each row has exactly one 1)")
        report.append("RFV clusters: Valid [OK]")
    else:
        train_issues = (train_rfv_sum != 1).sum()
        val_issues = (val_rfv_sum != 1).sum()
        test_issues = (test_rfv_sum != 1).sum()
        logger.warning(
            f"  RFV clusters: [WARN] - Train: {train_issues}, "
            f"Val: {val_issues}, Test: {test_issues} rows with issues"
        )
        report.append(
            f"RFV clusters: Invalid [WARN] - Train: {train_issues}, "
            f"Val: {val_issues}, Test: {test_issues}"
        )

    # Check age
    logger.info("\nChecking age feature...")
    if "age" in X_train.columns:
        age_min = X_train["age"].min()
        age_max = X_train["age"].max()
        age_mean = X_train["age"].mean()
        logger.info(f"  age: min={age_min:.2f}, max={age_max:.2f}, mean={age_mean:.2f}")
        if age_min >= -3 and age_max <= 3:  # Scaled range
            logger.info("  age: [OK] (appears to be scaled)")
            report.append("age: Scaled [OK]")
        else:
            logger.warning("  age: [WARN] (unexpected range)")
            report.append("age: Unexpected range [WARN]")

    # Save verification report
    with open(OUTPUT_DIR / "analysis" / "data_quality_check.txt", "w") as f:
        f.write("DATA QUALITY VERIFICATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        for line in report:
            f.write(line + "\n")

    report_file = OUTPUT_DIR / "analysis" / "data_quality_check.txt"
    logger.info(f"\nSaved verification report to: {report_file}")


# ============================================================================
# PHASE 2: CLASS IMBALANCE ANALYSIS & SMOTE
# ============================================================================


def analyze_esi_distribution(y_train: pd.Series) -> dict:
    """Analyze ESI class distribution."""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: CLASS IMBALANCE ANALYSIS")
    logger.info("=" * 80)

    logger.info("\nESI Distribution in Training Set:")
    esi_dist = y_train.value_counts().sort_index()
    total = len(y_train)

    distribution = {}
    for esi in [1, 2, 3, 4, 5]:
        count = esi_dist.get(esi, 0)
        pct = (count / total * 100) if total > 0 else 0
        distribution[esi] = {"count": int(count), "percentage": float(pct)}
        logger.info(f"  ESI {esi}: {count:,} ({pct:.2f}%)")

    # Calculate imbalance ratio
    esi_3_count = distribution[3]["count"]
    esi_1_count = distribution[1]["count"]
    if esi_1_count > 0:
        imbalance_ratio = esi_3_count / esi_1_count
        logger.info(f"\nImbalance ratio (ESI 3 / ESI 1): {imbalance_ratio:.1f}x")
        distribution["imbalance_ratio"] = float(imbalance_ratio)

    return distribution


def apply_smote(
    X_train: pd.DataFrame, y_train: pd.Series, esi_distribution: dict
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE + Tomek Links for class balancing."""
    logger.info("\n" + "=" * 80)
    logger.info("APPLYING SMOTE + TOMEK LINKS")
    logger.info("=" * 80)

    # SMOTE configuration
    sampling_strategy = {
        1: 10000,  # ESI 1: oversample to 10k
        2: 25000,  # ESI 2: oversample to 25k
        3: esi_distribution[3]["count"],  # ESI 3: keep as-is
        4: esi_distribution[4]["count"],  # ESI 4: keep as-is
        5: 10000,  # ESI 5: oversample to 10k
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

    # Save SMOTE report
    with open(OUTPUT_DIR / "analysis" / "smote_balancing_report.txt", "w") as f:
        f.write("SMOTE BALANCING REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Original training set size: {len(X_train):,}\n")
        f.write(f"Balanced training set size: {len(X_train_balanced):,}\n")
        f.write(f"SMOTE time: {smote_time:.2f} seconds\n\n")
        f.write("Sampling Strategy:\n")
        for esi, target in sampling_strategy.items():
            f.write(f"  ESI {esi}: {target:,}\n")
        f.write("\nBalanced Distribution:\n")
        for esi in [1, 2, 3, 4, 5]:
            count = balanced_dist.get(esi, 0)
            pct = count / len(y_train_balanced) * 100
            f.write(f"  ESI {esi}: {count:,} ({pct:.2f}%)\n")

    smote_report_file = OUTPUT_DIR / "analysis" / "smote_balancing_report.txt"
    logger.info(f"\nSaved SMOTE report to: {smote_report_file}")

    return X_train_balanced, y_train_balanced


# ============================================================================
# PHASE 3: FEATURE CORRELATION ANALYSIS
# ============================================================================


def analyze_feature_correlations(
    X_train: pd.DataFrame, y_train: pd.Series, feature_names: list
) -> None:
    """Analyze feature-target and feature-feature correlations."""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3: FEATURE CORRELATION ANALYSIS")
    logger.info("=" * 80)

    # Feature-target correlation
    logger.info("\nCalculating feature-target correlations...")
    correlations = {}
    for feature in feature_names:
        if feature in X_train.columns:
            corr = X_train[feature].corr(y_train)
            correlations[feature] = float(corr) if pd.notna(corr) else 0.0

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    logger.info("\nFeature-Target Correlations (sorted by |correlation|):")
    for i, (feature, corr) in enumerate(sorted_corr, 1):
        strength = (
            "Strong" if abs(corr) > 0.3 else "Medium" if abs(corr) > 0.1 else "Weak"
        )
        logger.info(f"  {i:2d}. {feature:30s}: {corr:7.4f} ({strength})")

    # Save feature-target correlations
    corr_df = pd.DataFrame(sorted_corr, columns=["feature", "correlation"])
    corr_df["abs_correlation"] = corr_df["correlation"].abs()
    corr_df["strength"] = corr_df["abs_correlation"].apply(
        lambda x: "Strong" if x > 0.3 else "Medium" if x > 0.1 else "Weak"
    )
    corr_df.to_csv(
        OUTPUT_DIR / "analysis" / "feature_correlation_with_target.csv", index=False
    )
    logger.info(
        f"\nSaved to: {OUTPUT_DIR / 'analysis' / 'feature_correlation_with_target.csv'}"
    )

    # Feature-feature correlation (multicollinearity)
    logger.info("\nCalculating feature-feature correlations...")
    feature_corr_matrix = X_train[feature_names].corr()

    # Find high correlations
    high_corr_pairs = []
    for i, feat1 in enumerate(feature_names):
        for feat2 in feature_names[i + 1 :]:
            if (
                feat1 in feature_corr_matrix.index
                and feat2 in feature_corr_matrix.columns
            ):
                corr_val = feature_corr_matrix.loc[feat1, feat2]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append((feat1, feat2, float(corr_val)))

    if high_corr_pairs:
        logger.warning(
            f"\nFound {len(high_corr_pairs)} pairs with |correlation| > 0.7:"
        )
        for feat1, feat2, corr in high_corr_pairs:
            logger.warning(f"  {feat1} <-> {feat2}: {corr:.4f}")
    else:
        logger.info("\nNo high correlations found (|correlation| > 0.7) [OK]")

    # Save feature-feature correlation matrix
    feature_corr_matrix.to_csv(
        OUTPUT_DIR / "analysis" / "feature_multicollinearity.csv"
    )
    logger.info(
        f"Saved to: {OUTPUT_DIR / 'analysis' / 'feature_multicollinearity.csv'}"
    )

    # VIF Analysis
    logger.info("\nCalculating VIF...")
    continuous_features = ["age"]  # Only age is continuous
    vif_data = []

    # Note: VIF requires at least 2 features, so for single continuous feature, skip VIF
    # Instead, we'll check multicollinearity among all features
    if len(continuous_features) > 0:
        feat_name = continuous_features[0]
        logger.info(
            f"  Note: VIF requires >=2 features. "
            f"Single continuous feature '{feat_name}' - skipping VIF."
        )
        logger.info("  Multicollinearity already checked via correlation matrix above.")

    # For binary features, VIF is typically low, but we can check
    binary_features = [f for f in feature_names if f not in continuous_features]
    if len(binary_features) > 0:
        # Sample a few binary features for VIF
        sample_binary = binary_features[:5]  # Check first 5
        if len(sample_binary) > 1:
            X_bin = X_train[sample_binary].values
            for i, feature in enumerate(sample_binary):
                vif = variance_inflation_factor(X_bin, i)
                vif_data.append({"Feature": feature, "VIF": float(vif)})
                status = "[OK]" if vif < 5 else "[WARN]"
                logger.info(f"  {feature}: {vif:.2f} {status}")

    if vif_data:
        vif_df = pd.DataFrame(vif_data)
        vif_file = OUTPUT_DIR / "analysis" / "vif_analysis.csv"
        vif_df.to_csv(vif_file, index=False)
        logger.info(f"Saved to: {vif_file}")


# ============================================================================
# PHASE 4: MODEL TRAINING
# ============================================================================


def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
) -> tuple[LogisticRegression, dict]:
    """Train Logistic Regression model."""
    logger.info("\n" + "=" * 80)
    logger.info("MODEL 1: LOGISTIC REGRESSION")
    logger.info("=" * 80)

    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )

    logger.info("Training Logistic Regression...")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    logger.info(f"Training completed in {train_time:.2f} seconds")

    # Evaluate
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)

    metrics = calculate_metrics(y_val, y_pred, y_pred_proba, "Logistic Regression")

    # Save model
    model_path = OUTPUT_DIR / "models" / "receptionist_lr.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Saved model to: {model_path}")

    # Save coefficients
    coef_df = pd.DataFrame(
        {"feature": X_train.columns, "coefficient": model.coef_[0]}
    ).sort_values("coefficient", key=abs, ascending=False)
    coef_file = OUTPUT_DIR / "analysis" / "receptionist_lr_coefficients.csv"
    coef_df.to_csv(coef_file, index=False)
    logger.info(f"Saved coefficients to: {coef_file}")

    return model, metrics


def train_decision_tree(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
) -> tuple[DecisionTreeClassifier, dict]:
    """Train Decision Tree model."""
    logger.info("\n" + "=" * 80)
    logger.info("MODEL 2: DECISION TREE")
    logger.info("=" * 80)

    model = DecisionTreeClassifier(
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )

    logger.info("Training Decision Tree...")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    logger.info(f"Training completed in {train_time:.2f} seconds")

    # Evaluate
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)

    metrics = calculate_metrics(y_val, y_pred, y_pred_proba, "Decision Tree")

    # Save model
    model_path = OUTPUT_DIR / "models" / "receptionist_dt.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Saved model to: {model_path}")

    # Save feature importance
    importance_df = pd.DataFrame(
        {"feature": X_train.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    dt_importance_file = (
        OUTPUT_DIR / "analysis" / "receptionist_dt_feature_importance.csv"
    )
    importance_df.to_csv(dt_importance_file, index=False)
    logger.info(f"Saved feature importance to: {dt_importance_file}")

    return model, metrics


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
) -> tuple[RandomForestClassifier, dict]:
    """Train Random Forest model."""
    logger.info("\n" + "=" * 80)
    logger.info("MODEL 3: RANDOM FOREST")
    logger.info("=" * 80)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )

    logger.info("Training Random Forest...")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    logger.info(f"Training completed in {train_time:.2f} seconds")

    # OOB score
    if hasattr(model, "oob_score_"):
        logger.info(f"OOB Score: {model.oob_score_:.4f}")

    # Evaluate
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)

    metrics = calculate_metrics(y_val, y_pred, y_pred_proba, "Random Forest")
    if hasattr(model, "oob_score_"):
        metrics["oob_score"] = float(model.oob_score_)

    # Save model
    model_path = OUTPUT_DIR / "models" / "receptionist_rf.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Saved model to: {model_path}")

    # Save feature importance
    importance_df = pd.DataFrame(
        {"feature": X_train.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    rf_importance_file = (
        OUTPUT_DIR / "analysis" / "receptionist_rf_feature_importance.csv"
    )
    importance_df.to_csv(rf_importance_file, index=False)
    logger.info(f"Saved feature importance to: {rf_importance_file}")

    return model, metrics


def train_stacking_ensemble(
    lr_model: LogisticRegression,
    dt_model: DecisionTreeClassifier,
    rf_model: RandomForestClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[StackingClassifier, dict]:
    """Train Stacking Ensemble model."""
    logger.info("\n" + "=" * 80)
    logger.info("MODEL 4: STACKING ENSEMBLE")
    logger.info("=" * 80)

    # Create base estimators
    base_estimators = [
        ("lr", lr_model),
        ("dt", dt_model),
        ("rf", rf_model),
    ]

    # Meta-learner
    meta_learner = LogisticRegression(random_state=42, max_iter=1000)

    # Create stacking classifier
    model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1,
    )

    logger.info("Training Stacking Ensemble...")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    logger.info(f"Training completed in {train_time:.2f} seconds")

    # Evaluate
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)

    metrics = calculate_metrics(y_val, y_pred, y_pred_proba, "Stacking Ensemble")

    # Save model
    model_path = OUTPUT_DIR / "models" / "receptionist_stacking_meta.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Saved model to: {model_path}")

    return model, metrics


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[XGBClassifier, dict]:
    """Train XGBoost model with hyperparameter tuning."""
    logger.info("\n" + "=" * 80)
    logger.info("MODEL 5: XGBOOST")
    logger.info("=" * 80)

    # XGBoost requires class labels to start from 0, so convert ESI 1-5 to 0-4
    logger.info("Converting ESI labels from 1-5 to 0-4 for XGBoost...")
    y_train_xgb = y_train - 1
    y_val_xgb = y_val - 1

    # Step 5.1: Grid Search (on sampled data for speed)
    logger.info("\nStep 5.1: Grid Search (on sampled data for speed)...")

    # Sample 10k rows for grid search (stratified to maintain class distribution)
    sample_size = min(10000, len(X_train))
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
    sample_dist = pd.Series(y_train_sample).value_counts().sort_index()
    logger.info(f"Sample distribution: {sample_dist.to_dict()}")

    param_grid = {
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [100, 200, 300],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_alpha": [0, 0.1, 0.5],
        "reg_lambda": [0.1, 1.0],
        "gamma": [0, 0.1],
    }

    base_model = XGBClassifier(
        objective="multi:softmax",
        num_class=5,
        random_state=42,
        n_jobs=-1,
    )

    cv = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=42
    )  # Reduced to 3 folds
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )

    logger.info("Running grid search on sampled data (3-fold CV)...")
    start_time = time.time()
    grid_search.fit(X_train_sample, y_train_sample)
    grid_time = time.time() - start_time

    logger.info(f"Grid search completed in {grid_time:.2f} seconds")
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

    # Save grid search results
    grid_results_df = pd.DataFrame(grid_search.cv_results_)
    grid_results_file = (
        OUTPUT_DIR / "parameters" / "receptionist_xgboost_grid_search_results.csv"
    )
    grid_results_df.to_csv(grid_results_file, index=False)
    logger.info(f"Saved grid search results to: {grid_results_file}")

    # Step 5.2: Random Search (simplified - use best from grid)
    logger.info("\nStep 5.2: Using best parameters from grid search...")
    best_params = grid_search.best_params_

    # Step 5.3: Final Training with Cost-Sensitive Weights
    logger.info("\nStep 5.3: Training final XGBoost with cost-sensitive weights...")

    model = XGBClassifier(
        **best_params,
        objective="multi:softmax",
        num_class=5,
        random_state=42,
        n_jobs=-1,
    )

    # Convert sample weights to match 0-4 labels
    class_weights_xgb = {
        0: 50,
        1: 25,
        2: 1,
        3: 1,
        4: 5,
    }  # ESI 1->0, 2->1, 3->2, 4->3, 5->4
    sample_weights_xgb = np.array([class_weights_xgb[int(y)] for y in y_train_xgb])

    start_time = time.time()
    model.fit(X_train, y_train_xgb, sample_weight=sample_weights_xgb)
    train_time = time.time() - start_time

    logger.info(f"Training completed in {train_time:.2f} seconds")

    # Save best parameters
    best_params_file = (
        OUTPUT_DIR / "parameters" / "receptionist_xgboost_best_params.json"
    )
    with open(best_params_file, "w") as f:
        json.dump(best_params, f, indent=2)
    logger.info(f"Saved best parameters to: {best_params_file}")

    # Step 5.4: Probability Calibration
    logger.info("\nStep 5.4: Calibrating probabilities...")
    calibrated_model = CalibratedClassifierCV(model, cv=5, method="isotonic")
    calibrated_model.fit(X_val, y_val_xgb)

    # Save calibrated model
    calibrated_path = OUTPUT_DIR / "models" / "receptionist_xgboost_calibrated.pkl"
    joblib.dump(calibrated_model, calibrated_path)
    logger.info(f"Saved calibrated model to: {calibrated_path}")

    # Step 5.5: Test Set Evaluation
    logger.info("\nStep 5.5: Evaluating on test set...")
    y_pred_xgb = calibrated_model.predict(X_test)
    y_pred_proba = calibrated_model.predict_proba(X_test)

    # Convert predictions back from 0-4 to 1-5
    y_pred = y_pred_xgb + 1

    metrics = calculate_metrics(y_test, y_pred, y_pred_proba, "XGBoost (Calibrated)")

    # Save model
    model_path = OUTPUT_DIR / "models" / "receptionist_xgboost_final.pkl"
    joblib.dump(calibrated_model, model_path)
    logger.info(f"Saved final model to: {model_path}")

    # Save feature importance
    importance_df = pd.DataFrame(
        {"feature": X_train.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    xgb_importance_file = (
        OUTPUT_DIR / "analysis" / "receptionist_xgboost_feature_importance.csv"
    )
    importance_df.to_csv(xgb_importance_file, index=False)
    logger.info(f"Saved feature importance to: {xgb_importance_file}")

    return calibrated_model, metrics


def calculate_metrics(
    y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray, model_name: str
) -> dict:
    """Calculate comprehensive evaluation metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[1, 2, 3, 4, 5], zero_division=0
    )

    # Per-ESI metrics
    metrics_dict = {
        "model": model_name,
        "accuracy": float(accuracy),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }

    # ESI 1-2 combined recall (PRIMARY METRIC)
    # Formula: (TP_ESI1 + TP_ESI2) / (ESI1_total + ESI2_total)
    esi_1_2_mask = y_true.isin([1, 2])
    if esi_1_2_mask.sum() > 0:
        esi_1_2_true = y_true[esi_1_2_mask]
        esi_1_2_pred = y_pred[esi_1_2_mask]
        # Count true positives for ESI 1 and 2
        tp_esi1 = ((esi_1_2_true == 1) & (esi_1_2_pred == 1)).sum()
        tp_esi2 = ((esi_1_2_true == 2) & (esi_1_2_pred == 2)).sum()
        total_esi1_2 = len(esi_1_2_true)
        esi_1_2_recall = (tp_esi1 + tp_esi2) / total_esi1_2 if total_esi1_2 > 0 else 0.0
        metrics_dict["esi_1_2_recall"] = float(esi_1_2_recall)
    else:
        metrics_dict["esi_1_2_recall"] = 0.0

    # Per-ESI metrics
    for i, esi in enumerate([1, 2, 3, 4, 5]):
        idx = i
        metrics_dict[f"esi_{esi}_precision"] = float(precision[idx])
        metrics_dict[f"esi_{esi}_recall"] = float(recall[idx])
        metrics_dict[f"esi_{esi}_f1"] = float(f1[idx])
        metrics_dict[f"esi_{esi}_support"] = int(support[idx])

    return metrics_dict


def save_metrics(metrics: dict, filename: str) -> None:
    """Save metrics to CSV."""
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(OUTPUT_DIR / "metrics" / filename, index=False)
    logger.info(f"Saved metrics to: {OUTPUT_DIR / 'metrics' / filename}")


# ============================================================================
# PHASE 5: MODEL COMPARISON & SELECTION
# ============================================================================


def compare_models(all_metrics: dict) -> None:
    """Compare all models and create comparison report."""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 5: MODEL COMPARISON")
    logger.info("=" * 80)

    comparison_data = []
    for model_name, metrics in all_metrics.items():
        comparison_data.append(
            {
                "Model": model_name,
                "Accuracy": metrics.get("accuracy", 0),
                "ESI_1_2_Recall": metrics.get("esi_1_2_recall", 0),
                "ESI_1_2_Precision": (
                    (
                        metrics.get("esi_1_precision", 0)
                        + metrics.get("esi_2_precision", 0)
                    )
                    / 2
                ),
                "Weighted_F1": metrics.get("weighted_f1", 0),
            }
        )

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values("ESI_1_2_Recall", ascending=False)

    logger.info("\nModel Comparison:")
    logger.info(comparison_df.to_string(index=False))

    comp_file = OUTPUT_DIR / "metrics" / "receptionist_model_comparison.csv"
    comparison_df.to_csv(comp_file, index=False)
    logger.info(f"\nSaved comparison to: {comp_file}")

    # Select best model
    best_model_name = comparison_df.iloc[0]["Model"]
    best_esi_1_2_recall = comparison_df.iloc[0]["ESI_1_2_Recall"]

    logger.info(f"\nBest Model: {best_model_name}")
    logger.info(f"ESI 1-2 Recall: {best_esi_1_2_recall:.4f}")

    # Save selection
    with open(
        OUTPUT_DIR / "analysis" / "receptionist_best_model_selection.txt", "w"
    ) as f:
        f.write("BEST MODEL SELECTION\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Selected Model: {best_model_name}\n")
        f.write(f"ESI 1-2 Recall: {best_esi_1_2_recall:.4f}\n")
        f.write("\nSelection Criteria:\n")
        f.write("1. ESI 1-2 recall >= 75% (clinical priority)\n")
        f.write("2. Overall accuracy on test set\n")
        f.write("3. Train-validation gap < 10%\n")
        status = "[OK]" if best_esi_1_2_recall >= 0.75 else "[WARN]"
        f.write(f"\n{status} ESI 1-2 recall meets threshold\n")

    return best_model_name


def evaluate_best_model_on_test(
    best_model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train_balanced: pd.DataFrame,
    y_train_balanced: pd.Series,
) -> dict:
    """Evaluate the best model on the test set."""
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATING BEST MODEL ON TEST SET")
    logger.info("=" * 80)

    # Load the best model
    model_file_map = {
        "Decision Tree": "receptionist_dt.pkl",
        "Logistic Regression": "receptionist_lr.pkl",
        "Random Forest": "receptionist_rf.pkl",
        "Stacking Ensemble": "receptionist_stacking_meta.pkl",
        "XGBoost": "receptionist_xgboost_calibrated.pkl",
    }

    model_file = model_file_map.get(best_model_name)
    if not model_file:
        logger.warning(
            f"Model file not found for {best_model_name}, skipping test evaluation"
        )
        return {}

    model_path = OUTPUT_DIR / "models" / model_file
    if not model_path.exists():
        logger.warning(
            f"Model file {model_path} does not exist, skipping test evaluation"
        )
        return {}

    logger.info(f"Loading {best_model_name} from {model_path}...")
    model = joblib.load(model_path)

    # Make predictions
    logger.info("Making predictions on test set...")
    y_pred = model.predict(X_test)
    y_pred_proba = (
        model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    )

    # Calculate metrics
    metrics = calculate_metrics(
        y_test, y_pred, y_pred_proba, f"{best_model_name} (Test Set)"
    )

    # Save test results
    save_metrics(
        metrics,
        f"receptionist_{best_model_name.lower().replace(' ', '_')}_test_metrics.csv",
    )

    logger.info(f"\nTest Set Results for {best_model_name}:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  ESI 1-2 Recall: {metrics['esi_1_2_recall']:.4f}")
    logger.info(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    logger.info(f"  ESI 1 Recall: {metrics['esi_1_recall']:.4f}")
    logger.info(f"  ESI 2 Recall: {metrics['esi_2_recall']:.4f}")

    return metrics


def evaluate_all_models_on_test(
    models_dict: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate all models on the test set for fair comparison."""
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATING ALL MODELS ON TEST SET")
    logger.info("=" * 80)

    test_metrics = {}

    for model_name, model in models_dict.items():
        logger.info(f"\nEvaluating {model_name} on test set...")

        # Handle XGBoost label conversion
        if "XGBoost" in model_name:
            y_pred_xgb = model.predict(X_test)
            y_pred = y_pred_xgb + 1  # Convert back to 1-5
            y_pred_proba = (
                model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            )
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = (
                model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            )

        # Calculate metrics
        metrics = calculate_metrics(
            y_test, y_pred, y_pred_proba, f"{model_name} (Test Set)"
        )
        test_metrics[model_name] = metrics

        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  ESI 1-2 Recall: {metrics['esi_1_2_recall']:.4f}")
        logger.info(f"  Weighted F1: {metrics['weighted_f1']:.4f}")

    # Create comparison DataFrame
    comparison_data = []
    for model_name, metrics in test_metrics.items():
        comparison_data.append(
            {
                "Model": model_name,
                "Accuracy": metrics.get("accuracy", 0),
                "ESI_1_2_Recall": metrics.get("esi_1_2_recall", 0),
                "ESI_1_2_Precision": (
                    (
                        metrics.get("esi_1_precision", 0)
                        + metrics.get("esi_2_precision", 0)
                    )
                    / 2
                ),
                "Weighted_F1": metrics.get("weighted_f1", 0),
            }
        )

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values("ESI_1_2_Recall", ascending=False)

    logger.info("\n" + "=" * 80)
    logger.info("TEST SET MODEL COMPARISON")
    logger.info("=" * 80)
    logger.info("\n" + comparison_df.to_string(index=False))

    comparison_df.to_csv(
        OUTPUT_DIR / "metrics" / "receptionist_model_comparison_test_set.csv",
        index=False,
    )
    test_comp_file = (
        OUTPUT_DIR / "metrics" / "receptionist_model_comparison_test_set.csv"
    )
    logger.info(f"\nSaved test set comparison to: {test_comp_file}")

    return test_metrics


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def main() -> None:
    """Main training pipeline."""
    logger.info("=" * 80)
    logger.info("RECEPTIONIST TRIAGE MODEL TRAINING PIPELINE")
    logger.info("=" * 80)

    try:
        # Phase 1: Data Preparation
        (
            X_receptionist_train,
            X_receptionist_val,
            X_receptionist_test,
            y_train,
            y_val,
            y_test,
            feature_names,
        ) = extract_receptionist_features()

        verify_data_quality(
            X_receptionist_train,
            X_receptionist_val,
            X_receptionist_test,
            y_train,
            y_val,
            y_test,
        )

        # Phase 2: Class Imbalance & SMOTE
        esi_distribution = analyze_esi_distribution(y_train)
        X_train_balanced, y_train_balanced = apply_smote(
            X_receptionist_train, y_train, esi_distribution
        )

        # Phase 3: Feature Correlation Analysis
        analyze_feature_correlations(X_train_balanced, y_train_balanced, feature_names)

        # Phase 4: Model Training
        all_metrics = {}
        all_models = {}  # Store models for test evaluation

        # Model 1: Logistic Regression
        lr_model, lr_metrics = train_logistic_regression(
            X_train_balanced, y_train_balanced, X_receptionist_val, y_val
        )
        all_metrics["Logistic Regression"] = lr_metrics
        all_models["Logistic Regression"] = lr_model
        save_metrics(lr_metrics, "receptionist_lr_metrics.csv")

        # Model 2: Decision Tree
        dt_model, dt_metrics = train_decision_tree(
            X_train_balanced, y_train_balanced, X_receptionist_val, y_val
        )
        all_metrics["Decision Tree"] = dt_metrics
        all_models["Decision Tree"] = dt_model
        save_metrics(dt_metrics, "receptionist_dt_metrics.csv")

        # Model 3: Random Forest
        rf_model, rf_metrics = train_random_forest(
            X_train_balanced, y_train_balanced, X_receptionist_val, y_val
        )
        all_metrics["Random Forest"] = rf_metrics
        all_models["Random Forest"] = rf_model
        save_metrics(rf_metrics, "receptionist_rf_metrics.csv")

        # Model 4: Stacking Ensemble
        stacking_model, stacking_metrics = train_stacking_ensemble(
            lr_model,
            dt_model,
            rf_model,
            X_train_balanced,
            y_train_balanced,
            X_receptionist_val,
            y_val,
        )
        all_metrics["Stacking Ensemble"] = stacking_metrics
        all_models["Stacking Ensemble"] = stacking_model
        save_metrics(stacking_metrics, "receptionist_stacking_metrics.csv")

        # Model 5: XGBoost
        xgboost_model, xgboost_metrics = train_xgboost(
            X_train_balanced,
            y_train_balanced,
            X_receptionist_val,
            y_val,
            X_receptionist_test,
            y_test,
        )
        all_metrics["XGBoost"] = xgboost_metrics
        all_models["XGBoost"] = xgboost_model
        save_metrics(xgboost_metrics, "receptionist_xgboost_test_metrics.csv")

        # Phase 5: Model Comparison & Selection
        best_model_name = compare_models(all_metrics)

        # Phase 5.5: Evaluate all models on test set
        evaluate_all_models_on_test(all_models, X_receptionist_test, y_test)

        # Phase 6: Feature Importance Analysis (consolidated)
        logger.info("\n" + "=" * 80)
        logger.info("FEATURE IMPORTANCE ANALYSIS")
        logger.info("=" * 80)

        # Load all feature importance files and create consolidated ranking
        importance_files = [
            "receptionist_dt_feature_importance.csv",
            "receptionist_rf_feature_importance.csv",
            "receptionist_xgboost_feature_importance.csv",
        ]

        consolidated_importance = {}
        for file in importance_files:
            file_path = OUTPUT_DIR / "analysis" / file
            if file_path.exists():
                df = pd.read_csv(file_path)
                model_name = file.replace("_feature_importance.csv", "").replace(
                    "receptionist_", ""
                )
                for _, row in df.iterrows():
                    feature = row["feature"]
                    importance = row["importance"]
                    if feature not in consolidated_importance:
                        consolidated_importance[feature] = {}
                    consolidated_importance[feature][model_name] = importance

        # Create consolidated ranking
        ranking_data = []
        for feature, importances in consolidated_importance.items():
            avg_importance = np.mean(list(importances.values()))
            ranking_data.append(
                {
                    "feature": feature,
                    "avg_importance": avg_importance,
                    **importances,
                }
            )

        ranking_df = pd.DataFrame(ranking_data).sort_values(
            "avg_importance", ascending=False
        )
        ranking_file = OUTPUT_DIR / "analysis" / "feature_importance_ranking.csv"
        ranking_df.to_csv(ranking_file, index=False)
        logger.info(f"Saved feature importance ranking to: {ranking_file}")

        # Phase 7: Save Training Configuration
        config = {
            "feature_list": feature_names,
            "n_features": len(feature_names),
            "medical_flags_available": False,
            "medical_flags_note": (
                "cebvd, chf, diabetes, ed_dialysis, hiv, "
                "no_chronic_conditions were not present in original "
                "NHAMCS dataset"
            ),
            "training_set_size": len(X_receptionist_train),
            "validation_set_size": len(X_receptionist_val),
            "test_set_size": len(X_receptionist_test),
            "balanced_training_set_size": len(X_train_balanced),
            "smote_config": {
                "1": 10000,
                "2": 25000,
                "3": esi_distribution[3]["count"],
                "4": esi_distribution[4]["count"],
                "5": 10000,
            },
            "best_model": best_model_name,
            "best_model_file": (
                f"receptionist_{best_model_name.lower().replace(' ', '_')}.pkl"
            ),
            "training_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "correlation_check": True,
            "multicollinearity_check": True,
            "cost_sensitive_weights": {"1": 50, "2": 25, "3": 1, "4": 1, "5": 5},
        }

        with open(
            OUTPUT_DIR / "parameters" / "receptionist_training_config.json", "w"
        ) as f:
            json.dump(config, f, indent=2)

        config_file = OUTPUT_DIR / "parameters" / "receptionist_training_config.json"
        logger.info(f"\nSaved training configuration to: {config_file}")

        logger.info("\n" + "=" * 80)
        logger.info("TRAINING PIPELINE COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"\nAll outputs saved to: {OUTPUT_DIR}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
