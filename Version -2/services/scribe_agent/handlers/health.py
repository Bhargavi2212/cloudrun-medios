"""
Health endpoint for the scribe-agent service.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from services.scribe_agent.core.soap import ScribeEngine

router = APIRouter(tags=["health"])


@router.get("/health", summary="Service health check")
async def health(request: Request) -> dict[str, str | bool]:
    """
    Return health information for monitoring with Gemini status.
    """
    engine: ScribeEngine = request.app.state.scribe_engine
    return {
        "status": "healthy",
        "service": "scribe-agent",
        "gemini_enabled": engine._enabled if hasattr(engine, "_enabled") else False,
        "gemini_initialized": engine._model is not None
        if hasattr(engine, "_model")
        else False,
        "has_api_key": bool(engine.gemini_api_key)
        if hasattr(engine, "gemini_api_key")
        else False,
    }
