"""
Federated training helper to generate model updates per hospital.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from federation.client import FederationClient
from federation.model_handler import (
    apply_logistic_weights,
    build_logistic_metadata,
    serialize_logistic_regression,
)
from federation.schemas import ModelUpdate
from services.manage_agent.config import ManageAgentSettings

logger = logging.getLogger(__name__)


class FederatedTrainer:
    """
    Executes a lightweight logistic regression training round and submits the update.
    """

    model_name = "manage-triage-logreg"

    def __init__(
        self,
        *,
        settings: ManageAgentSettings,
        federation_client: FederationClient | None,
    ) -> None:
        self.settings = settings
        self.federation_client = federation_client
        self.features_path = Path(settings.triage_features_path)
        self.labels_path = Path(settings.triage_labels_path)

    async def run_round(self) -> dict[str, Any]:
        """
        Train a model in a thread pool and submit the update asynchronously.
        """

        loop = asyncio.get_running_loop()
        global_model: dict[str, Any] | None = None
        if self.federation_client is not None:
            try:
                fetched = await self.federation_client.fetch_global_model(
                    self.model_name
                )
                global_model = fetched.model_dump()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != httpx.codes.NOT_FOUND:
                    raise
            except httpx.HTTPError:
                # Temporary loss of aggregator shouldn't fail the round.
                global_model = None

        payload = await loop.run_in_executor(
            None, partial(self._train_sync, global_model)
        )
        if self.federation_client is not None:
            update = ModelUpdate(**payload["model_update"])
            await self.federation_client.submit_update(update)
        return payload["summary"]

    def _train_sync(self, global_model: dict[str, Any] | None) -> dict[str, Any]:
        X = pd.read_csv(self.features_path)
        y = pd.read_csv(self.labels_path).squeeze()
        if "hospital_id" in X.columns:
            mask = X["hospital_id"] == self.settings.dol_hospital_id
            if mask.any():
                X = X.loc[mask]
                y = y.loc[mask]
            X = X.drop(columns=["hospital_id"])

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y if len(y.unique()) > 1 else None,
        )

        feature_names = list(X_train.columns)
        classes = list(np.unique(y_train))

        model = LogisticRegression(
            max_iter=1000,
            n_jobs=-1,
            multi_class="multinomial",
            solver="saga",
            warm_start=True,
        )
        self._initialize_model_metadata(model, feature_names, classes)

        baseline_accuracy: float | None = None
        if global_model is not None:
            metadata = global_model.get("metadata") or build_logistic_metadata(
                coef_shape=(len(classes), X_train.shape[1]),
                classes=classes,
                feature_names=feature_names,
            )
            apply_logistic_weights(
                model,
                weights=global_model["weights"],
                metadata=metadata,
                fallback_feature_names=feature_names,
            )
            baseline_accuracy = accuracy_score(y_val, model.predict(X_val))

        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)

        serialized_weights, metadata = serialize_logistic_regression(model)
        update_payload = {
            "model_name": self.model_name,
            "round_id": int(time.time()),
            "hospital_id": self.settings.dol_hospital_id,
            "weights": serialized_weights,
            "num_samples": int(len(X_train)),
            "metadata": metadata,
        }

        summary = {
            "hospital_id": self.settings.dol_hospital_id,
            "num_samples": int(len(X_train)),
            "accuracy": accuracy,
            "baseline_accuracy": baseline_accuracy,
        }
        logger.info(
            "Completed federated training round: accuracy=%.3f, samples=%s",
            accuracy,
            len(X_train),
        )
        return {"model_update": update_payload, "summary": summary}

    def _initialize_model_metadata(
        self,
        model: LogisticRegression,
        feature_names: Sequence[str],
        classes: Sequence[int | float | str],
    ) -> None:
        """
        Pre-populate sklearn attributes so we can seed coefficients before fitting.
        """

        model.classes_ = np.asarray(classes)
        model.n_features_in_ = len(feature_names)
        model.feature_names_in_ = np.asarray(feature_names)
