"""
Dependency helpers for federation aggregator.
"""

from __future__ import annotations

from fastapi import Request

from federation.config import AggregatorSettings
from federation.services.aggregator_service import AggregatorService


def get_settings(request: Request) -> AggregatorSettings:
    """
    Retrieve settings from application state.
    """

    return request.app.state.settings


def get_aggregator(request: Request) -> AggregatorService:
    """
    Retrieve the aggregator service from application state.
    """

    return request.app.state.aggregator
