"""
Helpers for serializing and deserializing supported model types.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


def build_logistic_metadata(
    *,
    coef_shape: tuple[int, int],
    classes: Sequence[int | float | str],
    feature_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Create metadata describing a logistic regression model.
    """

    return {
        "type": "logistic_regression",
        "coef_shape": list(coef_shape),
        "classes": list(classes),
        "feature_names": list(feature_names) if feature_names is not None else None,
    }


def serialize_logistic_regression(
    model: LogisticRegression,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """
    Serialize a logistic regression model into JSON-safe weights and metadata.
    """

    if not hasattr(model, "coef_") or not hasattr(model, "intercept_"):
        raise ValueError("Logistic regression model is not fitted.")

    weights = {
        "coef": model.coef_.ravel().tolist(),
        "intercept": model.intercept_.tolist(),
    }
    feature_names = getattr(model, "feature_names_in_", None)
    metadata = build_logistic_metadata(
        coef_shape=model.coef_.shape,
        classes=model.classes_,
        feature_names=feature_names,
    )
    return weights, metadata


def apply_logistic_weights(
    model: LogisticRegression,
    *,
    weights: dict[str, list[float]],
    metadata: dict[str, Any],
    fallback_feature_names: Sequence[str] | None = None,
) -> LogisticRegression:
    """
    Apply serialized weights to an unfitted logistic regression instance.
    """

    if metadata.get("type") != "logistic_regression":
        raise ValueError("Incompatible metadata type for logistic regression.")

    coef_shape = metadata.get("coef_shape")
    if coef_shape is None:
        raise ValueError("Coefficient shape missing from metadata.")

    coef_vector = np.asarray(weights["coef"], dtype=float)
    coef = coef_vector.reshape(tuple(coef_shape))
    intercept = np.asarray(weights["intercept"], dtype=float)

    model.classes_ = np.asarray(metadata.get("classes", []))
    if model.classes_.size == 0:
        raise ValueError("Class labels missing from metadata.")

    model.coef_ = coef
    model.intercept_ = intercept
    model.n_features_in_ = coef.shape[1]

    feature_names = metadata.get("feature_names") or fallback_feature_names
    if feature_names is not None:
        model.feature_names_in_ = np.array(feature_names)

    return model
