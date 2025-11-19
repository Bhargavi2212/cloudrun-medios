"""
Check-in service bridging manage-agent with the DOL portable profile endpoint.
"""

from __future__ import annotations

from uuid import UUID

import httpx

from services.manage_agent.schemas.portable_profile import PortableProfileResponse
from shared.exceptions import ConfigurationError


class CheckInService:
    """
    Fetch portable patient profiles from the DOL service.
    """

    def __init__(
        self, *, base_url: str | None, shared_secret: str | None, hospital_id: str
    ) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        self.shared_secret = shared_secret
        self.hospital_id = hospital_id

    async def fetch_profile(self, patient_id: UUID) -> PortableProfileResponse:
        """
        Retrieve the portable profile for the given patient identifier.
        """

        if not self.base_url or not self.shared_secret:
            raise ConfigurationError(
                "DOL integration is not configured for manage-agent."
            )

        headers = {
            "Authorization": f"Bearer {self.shared_secret}",
            "X-Requester": self.hospital_id,
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{self.base_url}/api/federated/patient",
                json={"patient_id": str(patient_id)},
                headers=headers,
            )
            response.raise_for_status()
            return PortableProfileResponse.model_validate(response.json())
