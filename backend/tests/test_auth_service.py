"""Tests for authentication service."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from backend.database.models import User, AccessToken
from backend.services.auth_service import AuthService
from backend.database.session import get_session
from backend.security.password import password_hasher


@pytest.fixture
def auth_service(db_session):
    """Create an auth service with test database session."""
    from contextlib import contextmanager
    
    @contextmanager
    def get_session_override():
        try:
            yield db_session
            # Commit changes made during the context
            db_session.commit()
        except Exception:
            db_session.rollback()
            raise
        # Don't close the session, it's managed by the fixture
    
    return AuthService(session_factory=get_session_override)


@pytest.fixture
def test_user_with_email(db_session, auth_service):
    """Create a test user with a specific email."""
    from backend.database.models import UserStatus, Role, UserRole
    
    # Create a role first
    role = Role(
        id=str(uuid4()),
        name="DOCTOR",
        description="Doctor role",
        is_deleted=False,
    )
    db_session.add(role)
    db_session.flush()
    
    # Create user
    user = User(
        id=str(uuid4()),
        email="auth_test@example.com",
        password_hash=password_hasher.hash("password123"),
        first_name="Auth",
        last_name="Test",
        status=UserStatus.ACTIVE,
        is_deleted=False,
    )
    db_session.add(user)
    db_session.flush()
    
    # Associate role with user
    user_role = UserRole(
        user_id=user.id,
        role_id=role.id,
    )
    db_session.add(user_role)
    db_session.commit()
    db_session.refresh(user)
    return user


def test_authenticate_user_success(auth_service, test_user_with_email):
    """Test successful user authentication."""
    access_token, refresh_token, user = auth_service.authenticate_user("auth_test@example.com", "password123")
    assert user is not None
    assert user.email == "auth_test@example.com"
    assert user.id == test_user_with_email.id
    assert access_token is not None
    assert refresh_token is not None


def test_authenticate_user_wrong_password(auth_service, test_user_with_email):
    """Test authentication with wrong password."""
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        auth_service.authenticate_user("auth_test@example.com", "wrongpassword")
    assert exc_info.value.status_code == 401
    assert "Invalid credentials" in str(exc_info.value.detail)


def test_authenticate_user_nonexistent(auth_service):
    """Test authentication with non-existent user."""
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        auth_service.authenticate_user("nonexistent@example.com", "password123")
    assert exc_info.value.status_code == 401
    assert "Invalid credentials" in str(exc_info.value.detail)


def test_request_password_reset(auth_service, test_user_with_email, db_session):
    """Test password reset token generation."""
    token = auth_service.request_password_reset("auth_test@example.com")
    assert token is not None
    assert len(token) > 0
    
    # Verify token was saved to database
    token_record = db_session.query(AccessToken).filter(
        AccessToken.purpose == "password_reset"
    ).first()
    assert token_record is not None
    assert token_record.user_id == test_user_with_email.id
    assert token_record.consumed_at is None


def test_request_password_reset_nonexistent_user(auth_service):
    """Test password reset request for non-existent user (should not raise error)."""
    token = auth_service.request_password_reset("nonexistent@example.com")
    # Should return empty string for security (don't reveal if email exists)
    assert token == ""


def test_reset_password_success(auth_service, test_user_with_email, db_session):
    """Test successful password reset."""
    # Request reset token
    reset_token = auth_service.request_password_reset("auth_test@example.com")
    assert reset_token is not None
    assert len(reset_token) > 0
    
    # Reset password
    auth_service.reset_password(reset_token, "newpassword123")
    
    # Verify password was changed - need to refresh the user from database
    db_session.expire_all()  # Expire all objects to force refresh
    db_session.refresh(test_user_with_email)
    
    # Verify the password hash was updated
    assert password_hasher.verify("newpassword123", test_user_with_email.password_hash)
    # Also verify old password doesn't work
    assert not password_hasher.verify("password123", test_user_with_email.password_hash)
    
    # Verify token was consumed
    db_session.expire_all()
    token_record = db_session.query(AccessToken).filter(
        AccessToken.purpose == "password_reset",
        AccessToken.consumed_at.isnot(None)
    ).first()
    assert token_record is not None


def test_reset_password_invalid_token(auth_service):
    """Test password reset with invalid token."""
    from fastapi import HTTPException
    
    with pytest.raises(HTTPException) as exc_info:
        auth_service.reset_password("invalid_token", "newpassword123")
    assert exc_info.value.status_code == 400
    assert "Invalid or expired reset token" in str(exc_info.value.detail)


def test_reset_password_expired_token(auth_service, test_user_with_email, db_session):
    """Test password reset with expired token."""
    # Request reset token
    reset_token = auth_service.request_password_reset("auth_test@example.com")
    
    # Manually expire the token
    token_record = db_session.query(AccessToken).filter(
        AccessToken.purpose == "password_reset"
    ).first()
    token_record.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
    db_session.commit()
    
    # Try to reset password (should fail)
    with pytest.raises(Exception):  # Should raise HTTPException
        auth_service.reset_password(reset_token, "newpassword123")


def test_reset_password_already_used_token(auth_service, test_user_with_email, db_session):
    """Test password reset with already used token."""
    from fastapi import HTTPException
    
    # Request reset token
    reset_token = auth_service.request_password_reset("auth_test@example.com")
    assert reset_token is not None
    
    # Use the token once
    auth_service.reset_password(reset_token, "newpassword123")
    
    # Flush/commit to ensure token is marked as consumed
    db_session.flush()
    db_session.commit()
    
    # Try to use it again (should fail)
    with pytest.raises(HTTPException) as exc_info:
        auth_service.reset_password(reset_token, "anotherpassword123")
    assert exc_info.value.status_code == 400
    assert "Invalid or expired reset token" in str(exc_info.value.detail)

