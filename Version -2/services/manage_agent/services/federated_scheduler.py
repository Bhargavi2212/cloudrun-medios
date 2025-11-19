"""
Background loop that periodically triggers federated training rounds.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any

import httpx

from federation.client import FederationClient
from services.manage_agent.core.triage import TriageEngine
from services.manage_agent.services.federated_trainer import FederatedTrainer

logger = logging.getLogger(__name__)


class FederatedTrainingLoop:
    """
    Coordinates asynchronous training rounds and refreshing local inference state.
    """

    def __init__(
        self,
        *,
        trainer: FederatedTrainer,
        triage_engine: TriageEngine,
        federation_client: FederationClient | None,
        interval_seconds: int,
    ) -> None:
        self._trainer = trainer
        self._triage_engine = triage_engine
        self._federation_client = federation_client
        self._interval = max(interval_seconds, 60)
        self._task: asyncio.Task[Any] | None = None

    def start(self) -> None:
        """
        Launch the background loop if it is not already running.
        """

        if self._task is None:
            self._task = asyncio.create_task(
                self._run_loop(), name="federated-training-loop"
            )
            logger.info(
                "🌀 Federated training loop started (interval=%ss)", self._interval
            )

    async def shutdown(self) -> None:
        """
        Stop the background loop gracefully.
        """

        if self._task is None:
            return
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task
        self._task = None
        logger.info("🛑 Federated training loop stopped.")

    async def _run_loop(self) -> None:
        """
        Periodically trigger a training round and refresh the cached global model.
        """

        consecutive_failures = 0
        max_consecutive_failures = 5

        while True:
            try:
                await self._execute_round()
                consecutive_failures = 0
            except asyncio.CancelledError:
                raise
            except httpx.HTTPError as exc:
                consecutive_failures += 1
                logger.error(
                    "Federated training round failed due to HTTP error (consecutive failures: %d/%d): %s",  # noqa: E501
                    consecutive_failures,
                    max_consecutive_failures,
                    exc,
                )
                if consecutive_failures >= max_consecutive_failures:
                    logger.critical(
                        "Too many consecutive failures (%d), pausing federated training for %d seconds",  # noqa: E501
                        consecutive_failures,
                        self._interval * 2,
                    )
                    await asyncio.sleep(self._interval * 2)
                    consecutive_failures = 0
            except Exception:  # pragma: no cover - defensive logging
                consecutive_failures += 1
                logger.exception(
                    "Federated training round failed with unexpected error (consecutive failures: %d/%d)",  # noqa: E501
                    consecutive_failures,
                    max_consecutive_failures,
                )
                if consecutive_failures >= max_consecutive_failures:
                    logger.critical(
                        "Too many consecutive failures (%d), pausing federated training for %d seconds",  # noqa: E501
                        consecutive_failures,
                        self._interval * 2,
                    )
                    await asyncio.sleep(self._interval * 2)
                    consecutive_failures = 0
            await asyncio.sleep(self._interval)

    async def _execute_round(self) -> None:
        """
        Run a single training round and then fetch the latest global model.
        """

        summary = await self._trainer.run_round()
        logger.info(
            "🏥 Federated round complete (hospital=%s, accuracy=%.3f, baseline=%s, samples=%s)",  # noqa: E501
            summary["hospital_id"],
            summary["accuracy"],
            summary["baseline_accuracy"],
            summary["num_samples"],
        )
        await self._refresh_global_model()

    async def _refresh_global_model(self) -> None:
        """
        Fetch the most recent global model and update the triage engine metadata.
        """

        if self._federation_client is None:
            return

        try:
            global_model = await self._federation_client.fetch_global_model(
                self._trainer.model_name
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != httpx.codes.NOT_FOUND:
                logger.warning("Failed to refresh global model: %s", exc)
            return
        except httpx.HTTPError as exc:
            logger.warning("Network error refreshing global model: %s", exc)
            return

        new_version = f"{self._triage_engine.model_version.split('::')[0]}::fed-round-{global_model.round_id}"  # noqa: E501
        self._triage_engine.update_model_version(new_version)
        logger.info(
            "📈 Updated triage engine model_version=%s (contributors=%s)",
            new_version,
            global_model.contributor_count,
        )
