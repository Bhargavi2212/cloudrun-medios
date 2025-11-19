"""
Dependency providers for the manage-agent service.
"""

from __future__ import annotations

from fastapi import Request

from services.manage_agent.config import ManageAgentSettings
from services.manage_agent.core.triage import TriageEngine
from services.manage_agent.services.check_in_service import CheckInService
from shared.config import get_settings as get_shared_settings


def get_triage_engine(request: Request) -> TriageEngine:
    """
    Retrieve the triage engine instance from application state.
    """

    engine: TriageEngine = request.app.state.triage_engine
    return engine


def get_check_in_service(request: Request) -> CheckInService:
    """
    Retrieve the check-in service configured on the application.
    """

    service: CheckInService = request.app.state.check_in_service
    return service


def get_settings() -> ManageAgentSettings:
    """
    Retrieve the manage-agent settings.
    """
    return get_shared_settings(ManageAgentSettings)
