"""
Model aggregation utilities implementing simple FedAvg.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from federation.schemas import GlobalModel, ModelUpdate


class AggregatorService:
    """
    Maintain federated learning state and perform FedAvg aggregation.
    """

    def __init__(self) -> None:
        self._updates: dict[tuple[str, int], list[ModelUpdate]] = defaultdict(list)
        self._global_models: dict[str, GlobalModel] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def submit_update(self, update: ModelUpdate) -> GlobalModel:
        """
        Store an update and recalculate the global model using FedAvg.
        """

        key = (update.model_name, update.round_id)
        self._updates[key].append(update)
        aggregated_weights, contributor_count = self._fed_avg(self._updates[key])

        metadata = update.metadata or self._metadata.get(update.model_name)
        if metadata is not None:
            self._metadata[update.model_name] = metadata

        global_model = GlobalModel(
            model_name=update.model_name,
            round_id=update.round_id,
            weights=aggregated_weights,
            contributor_count=contributor_count,
            metadata=metadata,
        )
        self._global_models[update.model_name] = global_model
        return global_model

    def get_global_model(self, model_name: str) -> GlobalModel | None:
        """
        Retrieve the latest aggregated model for the specified name.
        """

        return self._global_models.get(model_name)

    def _fed_avg(
        self, updates: list[ModelUpdate]
    ) -> tuple[dict[str, list[float]], int]:
        """
        Compute the sample-weighted average of supplied weight dictionaries.
        """

        if not updates:
            raise ValueError("No updates provided for aggregation.")

        first_update = updates[0]
        model_name = first_update.model_name
        round_id = first_update.round_id
        # Validate all updates are for the same model and round
        for update in updates[1:]:
            if update.model_name != model_name or update.round_id != round_id:
                raise ValueError(
                    f"All updates must have the same model_name and round_id. "
                    f"Expected ({model_name}, {round_id}), "
                    f"found ({update.model_name}, {update.round_id})"
                )
        layer_keys = first_update.weights.keys()
        total_samples = sum(update.num_samples for update in updates)
        if total_samples == 0:
            raise ValueError("Total number of samples must be greater than zero.")

        aggregated_weights: dict[str, list[float]] = {}
        for key in layer_keys:
            vector_length = len(updates[0].weights[key])
            weighted_sum = [0.0] * vector_length
            for update in updates:
                layer = update.weights[key]
                for idx in range(vector_length):
                    weighted_sum[idx] += layer[idx] * update.num_samples
            aggregated_weights[key] = [value / total_samples for value in weighted_sum]

        return aggregated_weights, len(updates)
