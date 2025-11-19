"""
Health endpoints for the manage-agent.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health", summary="Service health check")
async def health() -> dict[str, str]:
    """
    Return a simple health payload for monitoring integrations.
    """

    return {"status": "healthy", "service": "manage-agent"}
