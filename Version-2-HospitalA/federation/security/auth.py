"""
Authentication helper for federation aggregator.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status

from federation.config import AggregatorSettings
from federation.dependencies import get_settings


def verify_shared_secret(
    request: Request,
    settings: AggregatorSettings = Depends(get_settings),
) -> str:
    """
    Ensure the caller provides the configured shared secret.
    """

    auth_header = request.headers.get("Authorization")
    expected = f"Bearer {settings.shared_secret}"
    if auth_header != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized"
        )

    return request.headers.get("X-Hospital-ID", "unknown")
