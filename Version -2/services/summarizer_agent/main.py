"""
Application entry point for summarizer-agent.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

from services.summarizer_agent.config import SummarizerAgentSettings
from services.summarizer_agent.core.document_processor import DocumentProcessor
from services.summarizer_agent.core.summary import SummarizerEngine
from services.summarizer_agent.handlers import api_router
from shared.config import get_settings
from shared.fastapi import create_service_app

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all incoming requests for debugging."""

    async def dispatch(self, request: Request, call_next):
        import logging
        import sys

        logger = logging.getLogger(__name__)
        logger.info(
            "[REQUEST] INCOMING REQUEST: %s %s", request.method, request.url.path
        )
        print(
            f"[REQUEST] INCOMING REQUEST: {request.method} {request.url.path}",
            file=sys.stderr,
            flush=True,
        )
        try:
            response = await call_next(request)
            logger.info(
                "[REQUEST] RESPONSE: %s %s - Status %d",
                request.method,
                request.url.path,
                response.status_code,
            )
            print(
                f"[REQUEST] RESPONSE: {request.method} {request.url.path} - Status {response.status_code}",  # noqa: E501
                file=sys.stderr,
                flush=True,
            )
            return response
        except Exception as e:
            logger.error(
                "[REQUEST] ERROR in request: %s %s - %s",
                request.method,
                request.url.path,
                e,
                exc_info=True,
            )
            print(
                f"[ERROR] Error in request: {request.method} {request.url.path} - {e}",
                file=sys.stderr,
                flush=True,
            )
            raise


def create_app(settings: SummarizerAgentSettings | None = None) -> FastAPI:
    """
    Build the FastAPI application.
    """

    loaded_settings = settings or get_settings(SummarizerAgentSettings)

    # Log CORS configuration
    logger.info(f"CORS origins configured: {loaded_settings.cors_allow_origins}")

    app = create_service_app(
        service_name="summarizer-agent",
        version=loaded_settings.version,
        settings=loaded_settings,
        routers=[api_router],
        enable_database=True,
    )

    # Add request logging middleware AFTER CORS (so CORS is outermost)
    # Note: Middleware executes in reverse order, so this will be:
    # RequestLogging -> CORS -> handlers
    app.add_middleware(RequestLoggingMiddleware)

    app.state.settings = loaded_settings

    # Log Gemini API key status
    api_key = loaded_settings.gemini_api_key
    api_key_present = bool(api_key)
    api_key_length = len(api_key) if api_key else 0
    logger.info(f"Gemini API Key: present={api_key_present}, length={api_key_length}")
    if not api_key:
        logger.error("[CRITICAL] GEMINI_API_KEY is NOT set! Summarizer will use stub.")

    app.state.summarizer_engine = SummarizerEngine(
        model_version=loaded_settings.model_version,
        gemini_api_key=loaded_settings.gemini_api_key,
        gemini_model=loaded_settings.gemini_model,
    )

    # Verify engine initialization
    engine = app.state.summarizer_engine
    logger.info(
        f"SummarizerEngine initialized: enabled={engine._enabled}, model={engine._model is not None}"  # noqa: E501
    )

    app.state.document_processor = DocumentProcessor(
        api_key=loaded_settings.gemini_api_key,
        model_name=loaded_settings.gemini_model,
    )
    logger.info("Summarizer-agent app created and ready")
    return app


app = create_app()
