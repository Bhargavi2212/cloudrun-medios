"""
Application entry point for the scribe-agent service.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

from services.scribe_agent.config import ScribeAgentSettings
from services.scribe_agent.core.soap import ScribeEngine
from services.scribe_agent.handlers import api_router
from shared.config import get_settings
from shared.fastapi import create_service_app

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all incoming requests for debugging."""

    async def dispatch(self, request: Request, call_next):
        import sys

        print(
            f"[REQUEST] INCOMING: {request.method} {request.url.path}",
            file=sys.stderr,
            flush=True,
        )
        logger.info(
            "[REQUEST] Incoming request: %s %s", request.method, request.url.path
        )
        try:
            response = await call_next(request)
            print(
                f"[REQUEST] RESPONSE: {request.method} {request.url.path} - Status {response.status_code}",
                file=sys.stderr,
                flush=True,
            )
            logger.info(
                "[REQUEST] Response: %s %s - Status %d",
                request.method,
                request.url.path,
                response.status_code,
            )
            return response
        except Exception as e:
            print(
                f"[REQUEST] ERROR in request: {request.method} {request.url.path} - {e}",
                file=sys.stderr,
                flush=True,
            )
            logger.error(
                "[REQUEST] Error in request: %s %s - %s",
                request.method,
                request.url.path,
                e,
                exc_info=True,
            )
            raise


def create_app(settings: ScribeAgentSettings | None = None) -> FastAPI:
    """
    Build the FastAPI application for scribe-agent.
    """

    loaded_settings = settings or get_settings(ScribeAgentSettings)

    # Log CORS configuration
    logger.info(f"CORS origins configured: {loaded_settings.cors_allow_origins}")

    app = create_service_app(
        service_name="scribe-agent",
        version=loaded_settings.version,
        settings=loaded_settings,
        routers=[api_router],
        enable_database=True,
    )

    # Add request logging middleware AFTER CORS (so CORS is outermost)
    app.add_middleware(RequestLoggingMiddleware)

    app.state.settings = loaded_settings

    # HARD CHECK: Verify API Key is present
    api_key = loaded_settings.gemini_api_key
    if not api_key:
        logger.error(
            "[CRITICAL] GEMINI_API_KEY is NOT set in settings! Scribe agent will fail to generate AI notes."
        )
        # We won't raise an exception to allow startup, but we log visibly
    else:
        logger.info(
            "[INFO] GEMINI_API_KEY found in settings (length: %d)", len(api_key)
        )

    app.state.scribe_engine = ScribeEngine(
        model_version=loaded_settings.model_version,
        gemini_api_key=loaded_settings.gemini_api_key,
        gemini_model=loaded_settings.gemini_model,
    )
    logger.info("Scribe-agent app created and ready")
    return app


app = create_app()
