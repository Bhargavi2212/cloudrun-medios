"""
Client for fetching federated data from peer hospitals.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from typing import Any
from uuid import UUID

import httpx

from dol_service.config import PeerConfig

logger = logging.getLogger(__name__)


class PeerClient:
    """
    Fetch portable timeline fragments from peer DOL instances.
    """

    def __init__(
        self,
        peers: Iterable[PeerConfig],
        max_retries: int = 2,
        retry_delay: float = 0.5,
    ) -> None:
        self.peers = list(peers)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    async def fetch_profiles(self, patient_id: UUID) -> list[dict[str, Any]]:
        """
        Retrieve portable profiles from each configured peer.
        """

        if not self.peers:
            return []

        results: list[dict[str, Any]] = []
        async with httpx.AsyncClient(timeout=10.0) as client:
            for peer in self.peers:
                last_exception = None
                for attempt in range(self.max_retries + 1):
                    try:
                        headers = {}
                        if peer.api_key:
                            headers["Authorization"] = f"Bearer {peer.api_key}"
                        response = await client.post(
                            f"{peer.base_url}/api/federated/timeline",
                            json={"patient_id": str(patient_id)},
                            headers=headers,
                        )
                        response.raise_for_status()
                        payload = response.json()
                        if isinstance(payload, dict):
                            results.append(payload)
                        break
                    except (httpx.HTTPError, httpx.TimeoutException) as exc:
                        last_exception = exc
                        if attempt < self.max_retries:
                            delay = self.retry_delay * (2**attempt)
                            logger.debug(
                                "Peer fetch failed from %s (attempt %d/%d), "
                                "retrying in %.1fs: %s",
                                peer.name,
                                attempt + 1,
                                self.max_retries + 1,
                                delay,
                                exc,
                            )
                            await asyncio.sleep(delay)
                        else:
                            logger.warning(
                                "Failed to fetch peer data from %s after %d "
                                "attempts: %s",
                                peer.name,
                                self.max_retries + 1,
                                last_exception,
                            )
        return results
