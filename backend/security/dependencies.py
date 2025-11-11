"""FastAPI dependencies for authentication and authorization."""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Sequence
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session, joinedload

from ..database.models import User, UserStatus
from ..database.session import get_db_session
from .jwt import decode_access_token

bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    session: Session = Depends(get_db_session),
) -> User:
    """Get the current authenticated user.

    Returns 401 Unauthorized if no credentials are provided.
    Returns 401 Unauthorized if the token is invalid.
    Returns 403 Forbidden if the user is inactive.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    try:
        payload = decode_access_token(token)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user: User | None = (
        session.query(User).options(joinedload(User.roles)).filter(User.id == str(user_id), User.is_deleted.is_(False)).first()
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if user.status != UserStatus.ACTIVE:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User inactive")

    return user


def _normalize_roles(values: Sequence[Any]) -> set[str]:
    normalized: set[str] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            normalized.update(_normalize_roles(value))
            continue
        if isinstance(value, Enum):
            normalized.add(str(value.value))
        else:
            normalized.add(str(value))
    return normalized


def require_roles(*allowed_roles: Any) -> Callable[[User], User]:
    def dependency(user: User = Depends(get_current_user)) -> User:
        role_names = {role.name for role in user.roles or []}
        normalized_roles = _normalize_roles(allowed_roles)
        if normalized_roles and role_names.isdisjoint(normalized_roles):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
        return user

    return dependency
