"""
Application entry point for the manage-agent service.
"""

from __future__ import annotations

from fastapi import FastAPI

from services.manage_agent.config import ManageAgentSettings
from services.manage_agent.core.triage import TriageEngine
from services.manage_agent.handlers import api_router
from services.manage_agent.services.check_in_service import CheckInService
from shared.config import get_settings
from shared.fastapi import create_service_app


def create_app(settings: ManageAgentSettings | None = None) -> FastAPI:
    """
    Build and configure the FastAPI application.
    """

    loaded_settings = settings or get_settings(ManageAgentSettings)
    app = create_service_app(
        service_name="manage-agent",
        version=loaded_settings.version,
        settings=loaded_settings,
        routers=[api_router],
        enable_database=True,
    )
    app.state.settings = loaded_settings
    app.state.triage_engine = TriageEngine(model_version=loaded_settings.model_version)
    app.state.check_in_service = CheckInService(
        base_url=loaded_settings.dol_base_url,
        shared_secret=loaded_settings.dol_shared_secret,
        hospital_id=loaded_settings.dol_hospital_id,
    )
    return app


app = create_app()
