"""
HTTP client for interacting with the orchestrator/DOL service.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

import httpx

logger = logging.getLogger(__name__)


class OrchestratorClient:
    """
    Lightweight client that registers hospitals and pushes patient snapshots.
    """

    def __init__(
        self,
        *,
        base_url: str,
        shared_secret: str,
        hospital_id: str,
        timeout: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.shared_secret = shared_secret
        self.hospital_id = hospital_id
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.shared_secret}",
                "X-Requester": self.hospital_id,
                "Content-Type": "application/json",
            },
        )

    async def close(self) -> None:
        """
        Dispose of the underlying HTTP client.
        """

        await self._client.aclose()

    async def register(self, payload: dict[str, Any]) -> None:
        """
        Register or heartbeat this hospital with the orchestrator.
        """

        try:
            response = await self._client.post("/api/dol/registry", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - network failure
            logger.warning("Failed to register hospital with orchestrator: %s", exc)

    async def push_snapshot(self, patient_id: UUID, snapshot: dict[str, Any]) -> None:
        """
        Push a sanitized patient snapshot to the orchestrator.
        """

        try:
            response = await self._client.post(
                f"/api/dol/patients/{patient_id}/snapshot", json=snapshot
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - network failure
            logger.error("Failed to push patient snapshot for %s: %s", patient_id, exc)
