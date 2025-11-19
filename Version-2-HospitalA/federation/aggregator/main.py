"""
Aggregator service entry point.
"""

from __future__ import annotations

from fastapi import FastAPI

from federation.aggregator.handlers import api_router
from federation.config import AggregatorSettings
from federation.dependencies import get_settings as _get_settings  # noqa: F401
from federation.services.aggregator_service import AggregatorService
from shared.config import get_settings
from shared.fastapi import create_service_app


def create_app(settings: AggregatorSettings | None = None) -> FastAPI:
    """
    Construct the FastAPI application.
    """

    loaded_settings = settings or get_settings(AggregatorSettings)
    app = create_service_app(
        service_name="federation-aggregator",
        version=loaded_settings.version,
        settings=loaded_settings,
        routers=[api_router],
        enable_database=False,
    )
    app.state.settings = loaded_settings
    app.state.aggregator = AggregatorService()
    return app


app = create_app()
