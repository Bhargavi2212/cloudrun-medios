#!/usr/bin/env python3
"""Initialize model files if they don't exist (for Docker builds without local files)."""

# Helper classes must be defined at module level to be pickleable by joblib

from pathlib import Path

import joblib
import numpy as np


def create_models():
    """Create minimal working models if they don't exist."""

    model_files = [
        "xgboost_lightgbm_metadata.pkl",
        "baseline_metadata.pkl",
        "final_lightgbm_full_features.pkl",
        "final_xgboost_full_features.pkl",
        "final_stacking_ensemble.pkl",
    ]

    # Check if models already exist (copied from local)
    if all(Path(f).exists() for f in model_files[:2]):
        print("✓ Model files already exist, skipping generation")
        return

    print("Creating model files for deployment...")

    # 1. Metadata file
    metadata = {
        "feature_names": ["age", "systolic_bp", "diastolic_bp", "heart_rate", "temperature", "respiratory_rate"],
        "classes": [1, 2, 3, 4, 5],
        "severity_labels_0based": {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"},
        "model_version": "1.0.0",
        "created_at": "2024-01-01",
    }
    joblib.dump(metadata, "xgboost_lightgbm_metadata.pkl")
    print("[OK] Created xgboost_lightgbm_metadata.pkl")

    # 2. Baseline metadata
    baseline_metadata = {
        "feature_names": ["age", "systolic_bp", "diastolic_bp", "heart_rate", "temperature", "respiratory_rate"],
        "classes": [1, 2, 3, 4, 5],
        "severity_labels_0based": {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"},
    }
    joblib.dump(baseline_metadata, "baseline_metadata.pkl")
    print("[OK] Created baseline_metadata.pkl")

    model_payload = {
        "__mock_model__": True,
        "feature_names": metadata["feature_names"],
        "default_probs": [0.1, 0.1, 0.6, 0.1, 0.1],
    }

    joblib.dump(model_payload, "final_xgboost_full_features.pkl")
    print("[OK] Created final_xgboost_full_features.pkl (mock payload)")

    joblib.dump(model_payload, "final_lightgbm_full_features.pkl")
    print("[OK] Created final_lightgbm_full_features.pkl (mock payload)")

    joblib.dump(model_payload, "final_stacking_ensemble.pkl")
    print("[OK] Created final_stacking_ensemble.pkl (mock payload)")

    print("\n✓ All model files created successfully!")


if __name__ == "__main__":
    create_models()
