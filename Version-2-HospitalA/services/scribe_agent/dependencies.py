"""
Dependency providers for the scribe-agent service.
"""

from __future__ import annotations

from fastapi import Request

from services.scribe_agent.core.soap import ScribeEngine


def get_scribe_engine(request: Request) -> ScribeEngine:
    """
    Retrieve the ScribeEngine instance from application state.
    """

    engine: ScribeEngine = request.app.state.scribe_engine
    return engine
