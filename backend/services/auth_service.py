"""Authentication and authorization service."""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import jwt
from fastapi import HTTPException, status
from sqlalchemy.orm import Session, joinedload

from ..database import crud
from ..database.models import AccessToken, RefreshToken, Role, User, UserStatus
from ..database.session import get_session
from ..security.jwt import create_access_token, create_refresh_token
from ..security.password import password_hasher
from .config import get_settings

settings = get_settings()


class AuthService:
    def __init__(self, session_factory: Optional[Any] = None) -> None:
        """Initialize AuthService.

        Args:
            session_factory: Optional session factory for testing. If not provided,
                           uses the default get_session dependency.
        """
        self._session_factory = session_factory or get_session

    def register_user(
        self,
        email: str,
        password: str,
        first_name: Optional[str],
        last_name: Optional[str],
        role_names: List[str],
    ) -> User:
        email_normalized = email.lower()
        with self._session_factory() as session:
            existing = session.query(User).filter(User.email == email_normalized, User.is_deleted.is_(False)).first()
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Email already registered",
                )

            hashed_password = password_hasher.hash(password)
            user = crud.create_instance(
                session,
                User,
                email=email_normalized,
                password_hash=hashed_password,
                first_name=first_name,
                last_name=last_name,
                status=UserStatus.ACTIVE,
            )

            roles = self._lookup_roles(session, role_names)
            user.roles = roles
            session.flush()
            session.refresh(user)
            return user

    def authenticate_user(self, email: str, password: str) -> Tuple[str, str, User]:
        with self._session_factory() as session:
            user = (
                session.query(User)
                .options(joinedload(User.roles))
                .filter(User.email == email.lower(), User.is_deleted.is_(False))
                .first()
            )
            if not user or not password_hasher.verify(password, user.password_hash):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials",
                )

            if user.status != UserStatus.ACTIVE:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User inactive")

            user.last_login_at = datetime.now(timezone.utc)
            session.add(user)
            access_token = create_access_token(str(user.id), {"roles": [role.name for role in user.roles]})
            refresh_token, refresh_record = self._create_refresh_token_record(session, user)
            session.add(refresh_record)
            session.flush()
            return access_token, refresh_token, user

    def refresh_tokens(self, refresh_token: str) -> Tuple[str, str]:
        with self._session_factory() as session:
            try:
                payload = jwt.decode(
                    refresh_token,
                    settings.jwt_refresh_secret,
                    algorithms=[settings.jwt_algorithm],
                )
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid refresh token",
                ) from exc

            if payload.get("type") != "refresh":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type",
                )

            token_id = payload.get("jti")
            user_id = payload.get("sub")
            if not token_id or not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token payload",
                )

            hashed = self._hash_token(refresh_token)
            record = (
                session.query(RefreshToken)
                .filter(
                    RefreshToken.id == token_id,
                    RefreshToken.user_id == user_id,
                    RefreshToken.token_hash == hashed,
                    RefreshToken.revoked.is_(False),
                    RefreshToken.is_deleted.is_(False),
                )
                .first()
            )
            if not record:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Refresh token revoked",
                )

            expires_at = record.expires_at
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if expires_at < datetime.now(timezone.utc):
                record.revoked = True
                session.add(record)
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Refresh token expired",
                )

            user = session.query(User).options(joinedload(User.roles)).filter(User.id == user_id).first()
            if not user or user.status != UserStatus.ACTIVE:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User inactive")

            record.revoked = True
            session.add(record)

            access_token = create_access_token(str(user.id), {"roles": [role.name for role in user.roles]})
            new_refresh_token, new_record = self._create_refresh_token_record(session, user)
            session.add(new_record)
            session.flush()
            return access_token, new_refresh_token

    def revoke_refresh_token(self, refresh_token: str) -> None:
        with self._session_factory() as session:
            hashed = self._hash_token(refresh_token)
            record = (
                session.query(RefreshToken).filter(RefreshToken.token_hash == hashed, RefreshToken.revoked.is_(False)).first()
            )
            if record:
                record.revoked = True
                session.add(record)

    def update_user_profile(
        self,
        user_id: str,
        *,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        phone: Optional[str] = None,
    ) -> User:
        with self._session_factory() as session:
            user: Optional[User] = (
                session.query(User)
                .options(joinedload(User.roles))
                .filter(User.id == str(user_id), User.is_deleted.is_(False))
                .first()
            )
            if not user:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

            if first_name is not None:
                user.first_name = first_name
            if last_name is not None:
                user.last_name = last_name
            if phone is not None:
                user.phone = phone

            session.add(user)
            session.flush()
            session.refresh(user)
            return user

    def change_password(self, user_id: str, current_password: str, new_password: str) -> None:
        with self._session_factory() as session:
            user: Optional[User] = session.query(User).filter(User.id == str(user_id), User.is_deleted.is_(False)).first()
            if not user:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

            if not password_hasher.verify(current_password, user.password_hash):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Current password is incorrect",
                )

            user.password_hash = password_hasher.hash(new_password)
            user.must_reset_password = False
            session.add(user)

    def request_password_reset(self, email: str) -> str:
        """Generate a password reset token and return it.

        In a production environment, this token would be sent via email.
        For now, we return it so the frontend can use it for testing.
        """
        email_normalized = email.lower()
        with self._session_factory() as session:
            user: Optional[User] = (
                session.query(User).filter(User.email == email_normalized, User.is_deleted.is_(False)).first()
            )
            # Don't reveal if email exists (security best practice)
            if not user:
                return ""  # Return empty string, but don't raise error

            # Generate reset token
            reset_token = secrets.token_urlsafe(32)
            token_hash = self._hash_token(reset_token)
            expires_at = datetime.now(timezone.utc) + timedelta(hours=1)  # 1 hour expiry

            # Create access token record for password reset
            reset_token_record = AccessToken(
                user_id=str(user.id),
                purpose="password_reset",
                token_hash=token_hash,
                expires_at=expires_at,
                consumed_at=None,
            )
            session.add(reset_token_record)
            session.flush()

            # TODO: In production, send email with reset link
            # For now, return the token (frontend can use it for testing)
            return reset_token

    def reset_password(self, reset_token: str, new_password: str) -> None:
        """Reset password using a reset token."""
        token_hash = self._hash_token(reset_token)
        with self._session_factory() as session:
            # Find valid, unused reset token
            reset_token_record: Optional[AccessToken] = (
                session.query(AccessToken)
                .filter(
                    AccessToken.token_hash == token_hash,
                    AccessToken.purpose == "password_reset",
                    AccessToken.consumed_at.is_(None),
                    AccessToken.is_deleted.is_(False),
                )
                .first()
            )

            if not reset_token_record:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid or expired reset token",
                )

            # Check if token has expired
            expires_at = reset_token_record.expires_at
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if expires_at < datetime.now(timezone.utc):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Reset token has expired",
                )

            # Get user and update password
            user: Optional[User] = (
                session.query(User).filter(User.id == reset_token_record.user_id, User.is_deleted.is_(False)).first()
            )
            if not user:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

            # Update password
            user.password_hash = password_hasher.hash(new_password)
            user.must_reset_password = False
            session.add(user)

            # Mark token as consumed
            reset_token_record.consumed_at = datetime.now(timezone.utc)
            session.add(reset_token_record)

            # Revoke all refresh tokens for this user (security best practice)
            session.query(RefreshToken).filter(
                RefreshToken.user_id == str(user.id),
                RefreshToken.revoked.is_(False),
            ).update({RefreshToken.revoked: True})

    @staticmethod
    def serialize_user(user: User) -> Dict[str, Any]:
        roles = [role.name for role in user.roles] if user.roles else []
        primary_role = roles[0] if roles else "GUEST"
        full_name = " ".join(filter(None, [user.first_name, user.last_name])).strip()
        if not full_name:
            full_name = user.email
        return {
            "id": user.id,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "phone": user.phone,
            "status": user.status.value,
            "last_login_at": (user.last_login_at.isoformat() if user.last_login_at else None),
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "updated_at": user.updated_at.isoformat() if user.updated_at else None,
            "role": primary_role,
            "roles": roles,
            "full_name": full_name,
        }

    def _create_refresh_token_record(self, session: Session, user: User) -> Tuple[str, RefreshToken]:
        token_id = str(uuid4())
        refresh_token = create_refresh_token(str(user.id), token_id)
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_refresh_expires_minutes)
        hashed = self._hash_token(refresh_token)
        record = RefreshToken(
            id=token_id,
            user_id=str(user.id),
            token_hash=hashed,
            expires_at=expires_at,
            revoked=False,
        )
        return refresh_token, record

    def _lookup_roles(self, session: Session, role_names: List[str]) -> List[Role]:
        if not role_names:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one role required",
            )
        roles = session.query(Role).filter(Role.name.in_(role_names), Role.is_deleted.is_(False)).all()
        missing = set(role_names) - {role.name for role in roles}
        if missing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown roles: {', '.join(missing)}",
            )
        return roles

    @staticmethod
    def _hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()
