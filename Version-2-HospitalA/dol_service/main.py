"""
Application entry point for the Data Orchestration Layer.
"""

from __future__ import annotations

from fastapi import FastAPI

from dol_service.config import DOLSettings
from dol_service.dependencies import get_settings as _get_settings  # noqa: F401
from dol_service.handlers import api_router
from dol_service.services.peer_client import PeerClient
from shared.config import get_settings
from shared.fastapi import create_service_app


def create_app(settings: DOLSettings | None = None) -> FastAPI:
    """
    Instantiate the FastAPI application.
    """

    loaded_settings = settings or get_settings(DOLSettings)
    app = create_service_app(
        service_name="dol-service",
        version=loaded_settings.version,
        settings=loaded_settings,
        routers=[api_router],
        enable_database=True,
    )
    app.state.settings = loaded_settings
    app.state.peer_client = PeerClient(loaded_settings.peers)
    return app


app = create_app()
