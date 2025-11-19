"""
Summarizer-agent API routers.
"""

from fastapi import APIRouter

from services.summarizer_agent.handlers.documents import router as documents_router
from services.summarizer_agent.handlers.health import router as health_router
from services.summarizer_agent.handlers.summary import router as summary_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(summary_router)
api_router.include_router(documents_router)

__all__ = ["api_router"]
