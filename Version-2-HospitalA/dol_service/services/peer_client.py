"""
Client for fetching federated data from peer hospitals.
"""

from __future__ import annotations

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

    def __init__(self, peers: Iterable[PeerConfig]) -> None:
        self.peers = list(peers)

    async def fetch_profiles(self, patient_id: UUID) -> list[dict[str, Any]]:
        """
        Retrieve portable profiles from each configured peer.
        """

        if not self.peers:
            return []

        results: list[dict[str, Any]] = []
        async with httpx.AsyncClient(timeout=10.0) as client:
            for peer in self.peers:
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
                except httpx.HTTPError as exc:
                    logger.warning(
                        "⚠️  Failed to fetch peer data from %s: %s", peer.name, exc
                    )
        return results
