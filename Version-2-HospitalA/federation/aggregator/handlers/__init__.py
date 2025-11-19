"""
Aggregator service routers.
"""

from fastapi import APIRouter

from federation.aggregator.handlers.health import router as health_router
from federation.aggregator.handlers.model import router as model_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(model_router)

__all__ = ["api_router"]
