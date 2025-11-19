"""
Train a per-hospital triage model and export weights for federated learning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a hospital-specific triage model."
    )
    parser.add_argument(
        "--hospital-id",
        required=True,
        help="Identifier for the hospital running this training job.",
    )
    parser.add_argument(
        "--features",
        default="data/processed/X_train_final.csv",
        help="Path to the processed feature matrix CSV.",
    )
    parser.add_argument(
        "--labels",
        default="data/processed/y_train.csv",
        help="Path to the labels CSV (ESI scores).",
    )
    parser.add_argument(
        "--output",
        default="artifacts/triage_weights.json",
        help="Path where the serialized weight JSON will be written.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for validation.",
    )
    return parser.parse_args()


def load_dataset(
    features_path: Path, labels_path: Path
) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path).squeeze()
    return X, y


def filter_by_hospital(
    X: pd.DataFrame, y: pd.Series, hospital_id: str
) -> tuple[pd.DataFrame, pd.Series]:
    if "hospital_id" not in X.columns:
        return X, y
    mask = X["hospital_id"] == hospital_id
    if mask.sum() == 0:
        return X.drop(columns=["hospital_id"]), y
    return X.loc[mask].drop(columns=["hospital_id"]), y.loc[mask]


def train_model(
    X: pd.DataFrame, y: pd.Series, test_size: float
) -> tuple[LogisticRegression, float]:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    model = LogisticRegression(
        max_iter=1000, n_jobs=-1, multi_class="multinomial", solver="saga"
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    print("Validation accuracy:", accuracy)
    print(classification_report(y_val, predictions))
    return model, accuracy


def export_weights(
    model: LogisticRegression,
    feature_names: list[str],
    hospital_id: str,
    output_path: Path,
) -> None:
    payload = {
        "hospital_id": hospital_id,
        "model_name": "triage_logistic_regression",
        "num_features": len(feature_names),
        "weights": {
            "coef": model.coef_.tolist(),
            "intercept": model.intercept_.tolist(),
            "features": feature_names,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Exported weights to {output_path}")


if __name__ == "__main__":
    args = parse_args()
    features_path = Path(args.features)
    labels_path = Path(args.labels)
    output_path = Path(args.output)

    X, y = load_dataset(features_path, labels_path)
    X_filtered, y_filtered = filter_by_hospital(X, y, args.hospital_id)
    if "hospital_id" in X_filtered.columns:
        X_filtered = X_filtered.drop(columns=["hospital_id"])

    model, accuracy = train_model(X_filtered, y_filtered, args.test_size)
    export_weights(model, list(X_filtered.columns), args.hospital_id, output_path)
