"""
Dependency providers for the manage-agent service.
"""

from __future__ import annotations

from fastapi import Request

from services.manage_agent.config import ManageAgentSettings
from services.manage_agent.core.nurse_triage import NurseTriageEngine
from services.manage_agent.core.triage import TriageEngine
from services.manage_agent.services.check_in_service import CheckInService
from services.manage_agent.services.federated_sync_service import FederatedSyncService
from services.manage_agent.services.profile_merge_service import ProfileMergeService
from shared.config import get_settings as get_shared_settings


def get_triage_engine(request: Request) -> TriageEngine:
    """
    Retrieve the triage engine instance from application state.
    """

    engine: TriageEngine = request.app.state.triage_engine
    return engine


def get_nurse_triage_engine(request: Request) -> NurseTriageEngine:
    """
    Retrieve the nurse triage engine instance from application state.
    """

    engine: NurseTriageEngine = request.app.state.nurse_triage_engine
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


def get_federated_sync_service(request: Request) -> FederatedSyncService | None:
    """
    Retrieve the federated sync service if configured.
    """

    return getattr(request.app.state, "federated_sync_service", None)


def get_profile_merge_service(request: Request) -> ProfileMergeService:
    """
    Create a ProfileMergeService instance for the current request.
    """
    from database.session import get_session_factory

    settings = get_settings()
    session_factory = get_session_factory()
    session = session_factory()
    return ProfileMergeService(session=session, hospital_id=settings.dol_hospital_id)
