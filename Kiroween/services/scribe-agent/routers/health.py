"""
Health check endpoints for Scribe Agent service.
"""

from fastapi import APIRouter, HTTPException, status
from ..config import get_settings

router = APIRouter()


@router.get("/")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "service": "scribe-agent",
        "version": "2.0.0"
    }


@router.get("/detailed")
async def detailed_health_check():
    """Detailed health check including AI model availability."""
    settings = get_settings()
    
    # Check AI model availability
    ai_models_available = {
        "openai": bool(settings.openai_api_key),
        "gemini": bool(settings.gemini_api_key)
    }
    
    overall_status = "healthy" if any(ai_models_available.values()) else "degraded"
    
    return {
        "status": overall_status,
        "service": "scribe-agent",
        "version": "2.0.0",
        "hospital_id": settings.hospital_id,
        "ai_models": ai_models_available,
        "default_model": settings.default_model,
        "components": {
            "api": "healthy",
            "ai_models": "available" if any(ai_models_available.values()) else "unavailable"
        }
    }


@router.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe endpoint."""
    settings = get_settings()
    
    # Check if at least one AI model is configured
    if not settings.openai_api_key and not settings.gemini_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No AI models configured"
        )
    
    return {"status": "ready"}


@router.get("/live")
async def liveness_check():
    """Kubernetes liveness probe endpoint."""
    return {"status": "alive"}