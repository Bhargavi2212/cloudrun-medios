"""
Health endpoint for summarizer-agent.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health", summary="Service health check")
async def health() -> dict[str, str]:
    """
    Health status payload.
    """

    return {"status": "healthy", "service": "summarizer-agent"}
