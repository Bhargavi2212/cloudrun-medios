"""
Health endpoint for DOL.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api/dol", tags=["health"])


@router.get("/health", summary="Service health check")
async def health() -> dict[str, str]:
    """
    Return health status.
    """

    return {"status": "healthy", "service": "dol-service"}
