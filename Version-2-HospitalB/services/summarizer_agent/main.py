"""
Application entry point for summarizer-agent.
"""

from __future__ import annotations

from fastapi import FastAPI

from services.summarizer_agent.config import SummarizerAgentSettings
from services.summarizer_agent.core.document_processor import DocumentProcessor
from services.summarizer_agent.core.summary import SummarizerEngine
from services.summarizer_agent.handlers import api_router
from shared.config import get_settings
from shared.fastapi import create_service_app


def create_app(settings: SummarizerAgentSettings | None = None) -> FastAPI:
    """
    Build the FastAPI application.
    """

    loaded_settings = settings or get_settings(SummarizerAgentSettings)
    app = create_service_app(
        service_name="summarizer-agent",
        version=loaded_settings.version,
        settings=loaded_settings,
        routers=[api_router],
        enable_database=True,
    )
    app.state.settings = loaded_settings
    app.state.summarizer_engine = SummarizerEngine(
        model_version=loaded_settings.model_version
    )
    app.state.document_processor = DocumentProcessor(
        api_key=loaded_settings.gemini_api_key,
        model_name=loaded_settings.gemini_model,
    )
    return app


app = create_app()
