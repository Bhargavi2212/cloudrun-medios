"""
Health endpoint for the aggregator.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", summary="Aggregator health check")
async def health() -> dict[str, str]:
    """
    Return service health status.
    """

    return {"status": "healthy", "service": "federation-aggregator"}
