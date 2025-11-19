"""
Model aggregation utilities implementing simple FedAvg.
"""

from __future__ import annotations

from collections import defaultdict

from federation.schemas import GlobalModel, ModelUpdate


class AggregatorService:
    """
    Maintain federated learning state and perform FedAvg aggregation.
    """

    def __init__(self) -> None:
        self._updates: dict[tuple[str, int], list[ModelUpdate]] = defaultdict(list)
        self._global_models: dict[str, GlobalModel] = {}

    def submit_update(self, update: ModelUpdate) -> GlobalModel:
        """
        Store an update and recalculate the global model using FedAvg.
        """

        key = (update.model_name, update.round_id)
        self._updates[key].append(update)
        aggregated = self._fed_avg(self._updates[key])
        self._global_models[update.model_name] = aggregated
        return aggregated

    def get_global_model(self, model_name: str) -> GlobalModel | None:
        """
        Retrieve the latest aggregated model for the specified name.
        """

        return self._global_models.get(model_name)

    def _fed_avg(self, updates: list[ModelUpdate]) -> GlobalModel:
        """
        Compute the arithmetic mean of supplied weight dictionaries.
        """

        if not updates:
            raise ValueError("No updates provided for aggregation.")

        model_name = updates[0].model_name
        round_id = updates[0].round_id
        layer_keys = updates[0].weights.keys()

        aggregated_weights: dict[str, list[float]] = {}
        for key in layer_keys:
            layer_values = [update.weights[key] for update in updates]
            vector_length = len(layer_values[0])
            averaged_vector = [
                sum(values[i] for values in layer_values) / len(layer_values)
                for i in range(vector_length)
            ]
            aggregated_weights[key] = averaged_vector

        return GlobalModel(
            model_name=model_name,
            round_id=round_id,
            weights=aggregated_weights,
            contributor_count=len(updates),
        )
