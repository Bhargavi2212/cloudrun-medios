"""
Router assembly for the DOL service.
"""

from fastapi import APIRouter

from dol_service.handlers.federated import router as federated_router
from dol_service.handlers.health import router as health_router
from dol_service.handlers.patient_cache import router as patient_cache_router
from dol_service.handlers.registry import router as registry_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(federated_router)
api_router.include_router(registry_router)
api_router.include_router(patient_cache_router)

__all__ = ["api_router"]
