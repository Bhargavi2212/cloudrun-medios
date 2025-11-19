"""
Router assembly for the DOL service.
"""

from fastapi import APIRouter

from dol_service.handlers.federated import router as federated_router
from dol_service.handlers.health import router as health_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(federated_router)

__all__ = ["api_router"]
