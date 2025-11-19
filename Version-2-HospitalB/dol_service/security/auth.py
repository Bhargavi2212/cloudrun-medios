"""
Authentication utilities for federated requests.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status

from dol_service.config import DOLSettings
from dol_service.dependencies import get_settings


def verify_shared_secret(
    request: Request,
    settings: DOLSettings = Depends(get_settings),
) -> str:
    """
    Validate the shared secret provided by peer services.
    """

    auth_header = request.headers.get("Authorization")
    expected = f"Bearer {settings.shared_secret}"
    if auth_header != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized federated request.",
        )

    requester = request.headers.get("X-Requester", "peer")
    return requester
