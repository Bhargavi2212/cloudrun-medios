"""
Health check endpoints for Manage Agent service.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from shared.database import get_db_session
from config import get_settings

router = APIRouter()


@router.get("/")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "service": "manage-agent",
        "version": "2.0.0"
    }


@router.get("/detailed")
async def detailed_health_check(
    db: AsyncSession = Depends(get_db_session)
):
    """Detailed health check including database connectivity."""
    settings = get_settings()
    
    try:
        # Test database connection
        result = await db.execute(text("SELECT 1 as test"))
        db_healthy = result.scalar() == 1
        
        return {
            "status": "healthy" if db_healthy else "unhealthy",
            "service": "manage-agent",
            "version": "2.0.0",
            "hospital_id": settings.hospital_id,
            "database": "connected" if db_healthy else "disconnected",
            "components": {
                "database": "healthy" if db_healthy else "unhealthy",
                "api": "healthy"
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )


@router.get("/ready")
async def readiness_check(
    db: AsyncSession = Depends(get_db_session)
):
    """Kubernetes readiness probe endpoint."""
    try:
        # Verify database is accessible
        await db.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )


@router.get("/live")
async def liveness_check():
    """Kubernetes liveness probe endpoint."""
    return {"status": "alive"}