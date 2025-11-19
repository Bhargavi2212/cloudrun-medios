"""
Scribe-agent API routers.
"""

from fastapi import APIRouter

from services.scribe_agent.handlers.health import router as health_router
from services.scribe_agent.handlers.soap import router as soap_router
from services.scribe_agent.handlers.transcripts import router as transcripts_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(transcripts_router)
api_router.include_router(soap_router)

__all__ = ["api_router"]
