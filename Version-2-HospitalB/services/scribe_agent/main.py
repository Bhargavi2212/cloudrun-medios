"""
Application entry point for the scribe-agent service.
"""

from __future__ import annotations

from fastapi import FastAPI

from services.scribe_agent.config import ScribeAgentSettings
from services.scribe_agent.core.soap import ScribeEngine
from services.scribe_agent.handlers import api_router
from shared.config import get_settings
from shared.fastapi import create_service_app


def create_app(settings: ScribeAgentSettings | None = None) -> FastAPI:
    """
    Build the FastAPI application for scribe-agent.
    """

    loaded_settings = settings or get_settings(ScribeAgentSettings)
    app = create_service_app(
        service_name="scribe-agent",
        version=loaded_settings.version,
        settings=loaded_settings,
        routers=[api_router],
        enable_database=True,
    )
    app.state.settings = loaded_settings
    app.state.scribe_engine = ScribeEngine(model_version=loaded_settings.model_version)
    return app


app = create_app()
