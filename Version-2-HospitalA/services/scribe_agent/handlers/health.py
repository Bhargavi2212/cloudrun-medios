"""
Health endpoint for the scribe-agent service.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health", summary="Service health check")
async def health() -> dict[str, str]:
    """
    Return health information for monitoring.
    """

    return {"status": "healthy", "service": "scribe-agent"}
