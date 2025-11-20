"""
Application entry point for the scribe-agent service.
"""

from __future__ import annotations

import logging
import sys

from fastapi import FastAPI

from services.scribe_agent.config import ScribeAgentSettings
from services.scribe_agent.core.soap import ScribeEngine
from services.scribe_agent.handlers import api_router
from shared.config import get_settings
from shared.fastapi import create_service_app

logger = logging.getLogger(__name__)


def create_app(settings: ScribeAgentSettings | None = None) -> FastAPI:
    """
    Build the FastAPI application for scribe-agent.
    """

    try:
        loaded_settings = settings or get_settings(ScribeAgentSettings)
        logger.info("Creating FastAPI app for scribe-agent...")
        app = create_service_app(
            service_name="scribe-agent",
            version=loaded_settings.version,
            settings=loaded_settings,
            routers=[api_router],
            enable_database=True,
        )
        app.state.settings = loaded_settings
        logger.info("Initializing ScribeEngine...")
        app.state.scribe_engine = ScribeEngine(model_version=loaded_settings.model_version)
        logger.info("✅ Scribe-agent app created successfully")
        return app
    except Exception as e:
        logger.error("❌ Failed to create scribe-agent app: %s", e, exc_info=True)
        raise


try:
    app = create_app()
except Exception as e:
    print(f"FATAL: Failed to create application: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    raise
