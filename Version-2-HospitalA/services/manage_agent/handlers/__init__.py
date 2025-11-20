"""
Manage-agent API routers.
"""

from fastapi import APIRouter

from services.manage_agent.handlers.check_in import router as check_in_router
from services.manage_agent.handlers.documents import router as documents_router
from services.manage_agent.handlers.health import router as health_router
from services.manage_agent.handlers.patients import router as patients_router
from services.manage_agent.handlers.queue import router as queue_router
from services.manage_agent.handlers.triage import router as triage_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(patients_router)
api_router.include_router(triage_router)
api_router.include_router(check_in_router)
api_router.include_router(queue_router)
api_router.include_router(documents_router)

__all__ = ["api_router"]
