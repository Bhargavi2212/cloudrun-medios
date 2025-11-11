"""JWT helpers for access and refresh tokens."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import jwt

from ..services.config import get_settings

settings = get_settings()


def _current_time() -> datetime:
    return datetime.now(timezone.utc)


def create_access_token(sub: str, claims: Dict[str, Any] | None = None) -> str:
    to_encode: Dict[str, Any] = {
        "sub": sub,
        "type": "access",
        "iat": int(_current_time().timestamp()),
    }
    if claims:
        to_encode.update(claims)

    expire = _current_time() + timedelta(minutes=settings.jwt_access_expires_minutes)
    to_encode["exp"] = int(expire.timestamp())
    return jwt.encode(to_encode, settings.jwt_access_secret, algorithm=settings.jwt_algorithm)


def create_refresh_token(sub: str, token_id: str) -> str:
    expire = _current_time() + timedelta(minutes=settings.jwt_refresh_expires_minutes)
    payload = {
        "sub": sub,
        "type": "refresh",
        "iat": int(_current_time().timestamp()),
        "jti": token_id,
        "exp": int(expire.timestamp()),
    }
    return jwt.encode(payload, settings.jwt_refresh_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> Dict[str, Any]:
    return jwt.decode(token, settings.jwt_access_secret, algorithms=[settings.jwt_algorithm])


def decode_refresh_token(token: str) -> Dict[str, Any]:
    return jwt.decode(token, settings.jwt_refresh_secret, algorithms=[settings.jwt_algorithm])
