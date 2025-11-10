from fastapi import APIRouter

from . import auth, make_agent, manage_agent, patients, queue, summarizer, triage

api_router = APIRouter()
api_router.include_router(auth.router, tags=["auth"])
api_router.include_router(make_agent.router, prefix="/make-agent", tags=["make-agent"])
api_router.include_router(manage_agent.router, tags=["manage-agent"])
api_router.include_router(triage.router, tags=["triage"])
api_router.include_router(summarizer.router, tags=["summarizer"])
api_router.include_router(queue.router, tags=["queue"])
api_router.include_router(patients.router, tags=["patients"])

__all__ = ["api_router"]

