"""
Feature Engineering Module for Receptionist Triage Model

Creates additional features to improve model performance:
- Age bins (5 binary columns)
- RFV risk groups (3 binary columns)
- Interaction features (4 binary columns)
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_age_bins(X: pd.DataFrame, age_col: str = "age") -> pd.DataFrame:
    """
    Create age bin features from scaled age values.

    Uses percentiles to create bins that correspond to:
    - age_0_18: 0-18 years
    - age_19_35: 19-35 years
    - age_36_50: 36-50 years
    - age_51_65: 51-65 years
    - age_65_plus: 65+ years

    Args:
        X: DataFrame with age column (scaled)
        age_col: Name of age column

    Returns:
        DataFrame with 5 binary age bin columns
    """
    if age_col not in X.columns:
        logger.warning(f"Age column '{age_col}' not found. Skipping age bin creation.")
        return pd.DataFrame(index=X.index)

    age_values = X[age_col].values

    # Use percentiles to approximate age bins
    # Since age is scaled, we'll use percentiles of the distribution
    # These percentiles should roughly correspond to the age ranges
    p20 = np.percentile(age_values, 20)  # ~18 years
    p40 = np.percentile(age_values, 40)  # ~35 years
    p60 = np.percentile(age_values, 60)  # ~50 years
    p80 = np.percentile(age_values, 80)  # ~65 years

    age_bins = pd.DataFrame(index=X.index)

    # Create binary columns for each age bin
    age_bins["age_0_18"] = (age_values <= p20).astype(int)
    age_bins["age_19_35"] = ((age_values > p20) & (age_values <= p40)).astype(int)
    age_bins["age_36_50"] = ((age_values > p40) & (age_values <= p60)).astype(int)
    age_bins["age_51_65"] = ((age_values > p60) & (age_values <= p80)).astype(int)
    age_bins["age_65_plus"] = (age_values > p80).astype(int)

    logger.info(f"Created age bins: {age_bins.sum().to_dict()}")

    return age_bins


def create_rfv_risk_groups(X: pd.DataFrame) -> pd.DataFrame:
    """
    Create RFV risk group features.

    Groups RFV clusters into risk categories:
    - High risk: Neurological, Trauma_Injury, Respiratory
    - Medium risk: Fever_Infection, Gastrointestinal
    - Low risk: All other clusters

    Args:
        X: DataFrame with RFV cluster columns

    Returns:
        DataFrame with 3 binary risk group columns
    """
    rfv_cols = [col for col in X.columns if col.startswith("rfv1_cluster_")]

    if not rfv_cols:
        logger.warning(
            "No RFV cluster columns found. Skipping RFV risk group creation."
        )
        return pd.DataFrame(index=X.index)

    risk_groups = pd.DataFrame(index=X.index)

    # High risk clusters
    high_risk_clusters = [
        "rfv1_cluster_Neurological",
        "rfv1_cluster_Trauma_Injury",
        "rfv1_cluster_Respiratory",
    ]
    high_risk_cols = [col for col in high_risk_clusters if col in X.columns]
    if high_risk_cols:
        risk_groups["rfv_high_risk"] = X[high_risk_cols].max(axis=1).astype(int)
    else:
        risk_groups["rfv_high_risk"] = 0

    # Medium risk clusters
    medium_risk_clusters = [
        "rfv1_cluster_Fever_Infection",
        "rfv1_cluster_Gastrointestinal",
    ]
    medium_risk_cols = [col for col in medium_risk_clusters if col in X.columns]
    if medium_risk_cols:
        risk_groups["rfv_medium_risk"] = X[medium_risk_cols].max(axis=1).astype(int)
    else:
        risk_groups["rfv_medium_risk"] = 0

    # Low risk: all other clusters
    all_rfv_cols = set(rfv_cols)
    high_medium_cols = set(high_risk_cols + medium_risk_cols)
    low_risk_cols = list(all_rfv_cols - high_medium_cols)
    if low_risk_cols:
        risk_groups["rfv_low_risk"] = X[low_risk_cols].max(axis=1).astype(int)
    else:
        risk_groups["rfv_low_risk"] = 0

    logger.info(f"Created RFV risk groups: {risk_groups.sum().to_dict()}")

    return risk_groups


def create_interactions(X: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features.

    Creates 4 interaction features:
    - ambulance_arrival x age_65_plus
    - ambulance_arrival x rfv_high_risk
    - age_65_plus x injury
    - rfv_high_risk x injury

    Args:
        X: DataFrame with base features and engineered features

    Returns:
        DataFrame with 4 binary interaction columns
    """
    interactions = pd.DataFrame(index=X.index)

    # Check required columns exist
    required_cols = {
        "ambulance_arrival": "ambulance_arrival",
        "age_65_plus": "age_65_plus",
        "rfv_high_risk": "rfv_high_risk",
        "injury": "injury",
    }

    missing_cols = [name for name, col in required_cols.items() if col not in X.columns]
    if missing_cols:
        logger.warning(
            f"Missing columns for interactions: {missing_cols}. Creating zeros."
        )
        for col in [
            "interaction_ambulance_age65",
            "interaction_ambulance_rfv",
            "interaction_age65_injury",
            "interaction_rfv_injury",
        ]:
            interactions[col] = 0
        return interactions

    # Create interactions
    interactions["interaction_ambulance_age65"] = (
        X["ambulance_arrival"] * X["age_65_plus"]
    ).astype(int)

    interactions["interaction_ambulance_rfv"] = (
        X["ambulance_arrival"] * X["rfv_high_risk"]
    ).astype(int)

    interactions["interaction_age65_injury"] = (X["age_65_plus"] * X["injury"]).astype(
        int
    )

    interactions["interaction_rfv_injury"] = (X["rfv_high_risk"] * X["injury"]).astype(
        int
    )

    logger.info(f"Created interactions: {interactions.sum().to_dict()}")

    return interactions


def build_feature_set(
    X: pd.DataFrame, version: Literal["A", "B", "C", "D"] = "A"
) -> pd.DataFrame:
    """
    Build feature set based on version.

    Version A: Original 16 features
    Version B: Original + age bins (21 features)
    Version C: Original + age bins + RFV risk groups (24 features)
    Version D: Original + age bins + RFV risk groups + interactions (28 features)

    Args:
        X: DataFrame with original features
        version: Feature set version ('A', 'B', 'C', or 'D')

    Returns:
        DataFrame with selected features
    """
    logger.info(f"Building feature set version {version}...")

    # Start with original features
    original_features = ["age", "ambulance_arrival", "seen_72h", "injury"]
    rfv_features = [col for col in X.columns if col.startswith("rfv1_cluster_")]
    original_features.extend(sorted(rfv_features))

    # Verify all original features exist
    missing = [f for f in original_features if f not in X.columns]
    if missing:
        logger.warning(f"Missing original features: {missing}")
        original_features = [f for f in original_features if f in X.columns]

    X_selected = X[original_features].copy()

    if version == "A":
        logger.info(f"Feature set A: {len(X_selected.columns)} features")
        return X_selected

    # Version B: Add age bins
    if version in ["B", "C", "D"]:
        age_bins = create_age_bins(X, age_col="age")
        X_selected = pd.concat([X_selected, age_bins], axis=1)
        logger.info(f"Added age bins: {len(age_bins.columns)} features")

    if version == "B":
        logger.info(f"Feature set B: {len(X_selected.columns)} features")
        return X_selected

    # Version C: Add RFV risk groups
    if version in ["C", "D"]:
        rfv_risk_groups = create_rfv_risk_groups(X)
        X_selected = pd.concat([X_selected, rfv_risk_groups], axis=1)
        logger.info(f"Added RFV risk groups: {len(rfv_risk_groups.columns)} features")

    if version == "C":
        logger.info(f"Feature set C: {len(X_selected.columns)} features")
        return X_selected

    # Version D: Add interactions
    if version == "D":
        interactions = create_interactions(
            X_selected
        )  # Use X_selected which has age_65_plus and rfv_high_risk
        X_selected = pd.concat([X_selected, interactions], axis=1)
        logger.info(f"Added interactions: {len(interactions.columns)} features")
        logger.info(f"Feature set D: {len(X_selected.columns)} features")
        return X_selected

    return X_selected
