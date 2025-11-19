"""
Async client for interacting with the federation aggregator.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from federation.schemas import GlobalModel, ModelUpdate, ModelUpdateAck

logger = logging.getLogger(__name__)


class FederationClient:
    """
    Lightweight HTTP client for submitting updates to the aggregator.
    """

    def __init__(
        self,
        *,
        base_url: str,
        shared_secret: str,
        hospital_id: str,
        timeout: float = 10.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.shared_secret = shared_secret
        self.hospital_id = hospital_id
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    async def submit_update(self, update: ModelUpdate) -> ModelUpdateAck:
        """
        Submit a model update and return the acknowledgement.
        """

        payload = update.model_copy()
        if payload.hospital_id == "unknown":
            payload = payload.model_copy(update={"hospital_id": self.hospital_id})

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.base_url}/federation/submit",
                        headers=self._headers(),
                        json=payload.model_dump(),
                    )
                    response.raise_for_status()
                    return ModelUpdateAck.model_validate(response.json())
            except (
                httpx.HTTPStatusError,
                httpx.RequestError,
                httpx.TimeoutException,
            ) as exc:
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(
                        "Federation submit failed (attempt %d/%d), "
                        "retrying in %.1fs: %s",
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "Federation submit failed after %d attempts",
                        self.max_retries + 1,
                    )
                    raise

    async def fetch_global_model(self, model_name: str) -> GlobalModel:
        """
        Retrieve the aggregated global model.
        """

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.get(
                        f"{self.base_url}/federation/global-model/{model_name}",
                        headers=self._headers(include_hospital=False),
                    )
                    response.raise_for_status()
                    return GlobalModel.model_validate(response.json())
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == httpx.codes.NOT_FOUND:
                    raise
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(
                        "Federation fetch failed (attempt %d/%d), "
                        "retrying in %.1fs: %s",
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "Federation fetch failed after %d attempts",
                        self.max_retries + 1,
                    )
                    raise
            except (httpx.RequestError, httpx.TimeoutException) as exc:
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(
                        "Federation fetch failed (attempt %d/%d), "
                        "retrying in %.1fs: %s",
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "Federation fetch failed after %d attempts",
                        self.max_retries + 1,
                    )
                    raise

    def _headers(self, include_hospital: bool = True) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.shared_secret}"}
        if include_hospital:
            headers["X-Hospital-ID"] = self.hospital_id
        return headers
