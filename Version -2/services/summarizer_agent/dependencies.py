"""
Dependency providers for summarizer-agent.
"""

from __future__ import annotations

from fastapi import Request

from services.summarizer_agent.config import SummarizerAgentSettings
from services.summarizer_agent.core.document_processor import DocumentProcessor
from services.summarizer_agent.core.summary import SummarizerEngine
from shared.config import get_settings as get_shared_settings


def get_summarizer_engine(request: Request) -> SummarizerEngine:
    """
    Retrieve the summarizer engine from application state.
    """

    engine: SummarizerEngine = request.app.state.summarizer_engine
    return engine


def get_document_processor(request: Request) -> DocumentProcessor:
    """
    Retrieve the document processor from application state.
    """

    processor: DocumentProcessor = request.app.state.document_processor
    return processor


def get_settings() -> SummarizerAgentSettings:
    """
    Retrieve the summarizer-agent settings.
    """
    return get_shared_settings(SummarizerAgentSettings)
