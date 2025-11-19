"""
Check-in service bridging manage-agent with the DOL portable profile endpoint.
"""

from __future__ import annotations

import logging
from uuid import UUID

import httpx

from services.manage_agent.schemas.portable_profile import PortableProfileResponse
from shared.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class CheckInService:
    """
    Fetch portable patient profiles from the DOL service.
    Supports both patient_id and MRN-based lookups.
    """

    def __init__(
        self, *, base_url: str | None, shared_secret: str | None, hospital_id: str
    ) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        self.shared_secret = shared_secret
        self.hospital_id = hospital_id

    async def fetch_profile(
        self, patient_id: UUID, mrn: str | None = None
    ) -> PortableProfileResponse:
        """
        Retrieve the portable profile for the given patient identifier.
        Tries patient_id first, then falls back to MRN-based search if patient_id lookup fails.

        Args:
            patient_id: The patient UUID to search for
            mrn: Optional MRN to use for fallback matching if patient_id lookup fails
        """

        if not self.base_url or not self.shared_secret:
            raise ConfigurationError(
                "DOL integration is not configured for manage-agent."
            )

        # Try patient_id lookup first
        try:
            return await self._fetch_profile_by_patient_id(patient_id)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                # Patient not found by patient_id - try MRN matching if available
                if mrn:
                    logger.info(
                        "Patient %s not found in DOL by patient_id, trying MRN-based matching: %s",
                        patient_id,
                        mrn,
                    )
                    profile = await self.fetch_profile_by_mrn(mrn)
                    if profile:
                        logger.info(
                            "Found patient profile in DOL via MRN matching (MRN: %s, DOL patient_id: %s)",
                            mrn,
                            profile.patient.id,
                        )
                        return profile
                    logger.info(
                        "Patient not found in DOL by MRN %s either, no cross-hospital history available",
                        mrn,
                    )
                else:
                    logger.info(
                        "Patient %s not found in DOL by patient_id, and no MRN provided for fallback matching",
                        patient_id,
                    )
                # Re-raise the 404 if MRN matching also failed or wasn't attempted
                raise
            raise
        except Exception as e:
            logger.warning(
                "Failed to fetch profile by patient_id %s: %s", patient_id, e
            )
            raise

    async def find_patient_id_by_mrn(self, mrn: str) -> UUID | None:
        """
        Find patient_id (UUID) in DOL by MRN for cross-hospital patient matching.
        Returns None if patient not found.
        """
        if not self.base_url or not self.shared_secret:
            raise ConfigurationError(
                "DOL integration is not configured for manage-agent."
            )

        if not mrn:
            return None

        headers = {
            "Authorization": f"Bearer {self.shared_secret}",
            "X-Requester": self.hospital_id,
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}/api/dol/patients/search",
                    params={"mrn": mrn},
                    headers=headers,
                )
                if response.status_code == 404:
                    logger.info("Patient with MRN %s not found in DOL", mrn)
                    return None
                response.raise_for_status()
                data = response.json()
                patient_id_str = data.get("patient_id")
                if patient_id_str:
                    logger.info(
                        "Found patient_id %s for MRN %s in DOL", patient_id_str, mrn
                    )
                    return UUID(patient_id_str)
                return None
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.info("Patient with MRN %s not found in DOL", mrn)
                return None
            logger.warning("Failed to search DOL by MRN %s: %s", mrn, e)
            return None
        except Exception as e:
            logger.warning("Error searching DOL by MRN %s: %s", mrn, e)
            return None

    async def fetch_profile_by_mrn(self, mrn: str) -> PortableProfileResponse | None:
        """
        Retrieve the portable profile by MRN.
        First searches DOL for patient_id by MRN, then fetches the profile.
        Returns None if patient not found.
        """
        if not self.base_url or not self.shared_secret:
            raise ConfigurationError(
                "DOL integration is not configured for manage-agent."
            )

        if not mrn:
            return None

        # First, find patient_id by MRN
        patient_id = await self.find_patient_id_by_mrn(mrn)
        if patient_id is None:
            logger.info(
                "Patient with MRN %s not found in DOL, cannot fetch profile", mrn
            )
            return None

        # Then fetch profile using the found patient_id
        try:
            return await self._fetch_profile_by_patient_id(patient_id)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.info(
                    "Profile not found for patient_id %s (found via MRN %s)",
                    patient_id,
                    mrn,
                )
                return None
            raise
        except Exception as e:
            logger.warning(
                "Failed to fetch profile for patient_id %s (found via MRN %s): %s",
                patient_id,
                mrn,
                e,
            )
            raise

    async def _fetch_profile_by_patient_id(
        self, patient_id: UUID
    ) -> PortableProfileResponse:
        """
        Internal method to fetch profile by patient_id from DOL.
        """
        headers = {
            "Authorization": f"Bearer {self.shared_secret}",
            "X-Requester": self.hospital_id,
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            # First attempt to use the cached profile endpoint.
            try:
                cache_response = await client.get(
                    f"{self.base_url}/api/dol/patients/{patient_id}/profile",
                    headers=headers,
                )
                if cache_response.status_code == httpx.codes.OK:
                    logger.info(
                        "Retrieved profile from DOL cache for patient %s", patient_id
                    )
                    return PortableProfileResponse.model_validate(cache_response.json())
                if cache_response.status_code not in (
                    httpx.codes.NOT_FOUND,
                    httpx.codes.METHOD_NOT_ALLOWED,
                ):
                    cache_response.raise_for_status()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    logger.info(
                        "Patient %s not found in DOL cache, trying federated endpoint",
                        patient_id,
                    )
                else:
                    raise

            # Fallback to federated endpoint
            try:
                response = await client.post(
                    f"{self.base_url}/api/federated/patient",
                    json={"patient_id": str(patient_id)},
                    headers=headers,
                )
                response.raise_for_status()
                logger.info(
                    "Retrieved profile from DOL federated endpoint for patient %s",
                    patient_id,
                )
                return PortableProfileResponse.model_validate(response.json())
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    logger.info(
                        "Patient %s not found in DOL federated endpoint", patient_id
                    )
                raise
