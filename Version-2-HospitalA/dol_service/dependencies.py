"""
Dependency helpers for the DOL service.
"""

from __future__ import annotations

from fastapi import Depends, Request

from dol_service.config import DOLSettings
from dol_service.services.peer_client import PeerClient


def get_settings(request: Request) -> DOLSettings:
    """
    Retrieve settings from application state.
    """

    return request.app.state.settings


def get_peer_client(request: Request) -> PeerClient:
    """
    Retrieve the peer client from application state.
    """

    return request.app.state.peer_client


def get_hospital_id(settings: DOLSettings = Depends(get_settings)) -> str:
    """
    Provide the local hospital identifier.
    """

    return settings.hospital_id
