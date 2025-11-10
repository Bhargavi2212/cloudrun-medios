"""Tests for authentication API endpoints."""

from __future__ import annotations

from uuid import uuid4

import pytest

from backend.database.models import User
from backend.security.password import password_hasher


def test_login_success(client, test_user):
    """Test successful login."""
    response = client.post(
        "/api/v1/auth/login",
        json={"email": test_user.email, "password": "testpassword123"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "access_token" in data["data"]
    assert "refresh_token" in data["data"]


def test_login_wrong_password(client, test_user):
    """Test login with wrong password."""
    response = client.post(
        "/api/v1/auth/login",
        json={"email": test_user.email, "password": "wrongpassword"},
    )

    assert response.status_code == 401
    data = response.json()
    assert data["success"] is False


def test_login_nonexistent_user(client):
    """Test login with non-existent user."""
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "nonexistent@example.com", "password": "password123"},
    )

    assert response.status_code == 401
    data = response.json()
    assert data["success"] is False


def test_get_current_user(client, test_user, auth_headers):
    """Test getting current user info."""
    response = client.get("/api/v1/auth/me", headers=auth_headers)

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["email"] == test_user.email
    assert data["data"]["id"] == test_user.id


def test_get_current_user_unauthorized(client):
    """Test getting current user without authentication."""
    response = client.get("/api/v1/auth/me")

    assert response.status_code == 401


def test_forgot_password(client, test_user):
    """Test forgot password endpoint."""
    response = client.post(
        "/api/v1/auth/forgot-password",
        json={"email": test_user.email},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    # In development, token is returned in response
    if "reset_token" in (data.get("data") or {}):
        assert len(data["data"]["reset_token"]) > 0


def test_forgot_password_nonexistent_user(client):
    """Test forgot password for non-existent user (should still return success)."""
    response = client.post(
        "/api/v1/auth/forgot-password",
        json={"email": "nonexistent@example.com"},
    )

    # Should return success for security (don't reveal if email exists)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_reset_password(client, test_user, db_session):
    """Test reset password endpoint."""
    from contextlib import contextmanager

    from backend.services.auth_service import AuthService

    # Create auth service to generate reset token
    @contextmanager
    def get_session_override():
        try:
            yield db_session
        finally:
            pass

    auth_service = AuthService(session_factory=get_session_override)
    reset_token = auth_service.request_password_reset(test_user.email)

    # Reset password
    response = client.post(
        "/api/v1/auth/reset-password",
        json={
            "reset_token": reset_token,
            "new_password": "newpassword123",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True

    # Verify password was changed by trying to login with new password
    login_response = client.post(
        "/api/v1/auth/login",
        json={"email": test_user.email, "password": "newpassword123"},
    )
    assert login_response.status_code == 200


def test_reset_password_invalid_token(client):
    """Test reset password with invalid token."""
    response = client.post(
        "/api/v1/auth/reset-password",
        json={
            "reset_token": "invalid_token",
            "new_password": "newpassword123",
        },
    )

    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False


def test_change_password(client, test_user, auth_headers):
    """Test changing password."""
    response = client.post(
        "/api/v1/auth/change-password",
        json={
            "current_password": "testpassword123",
            "new_password": "newpassword123",
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True

    # Verify password was changed
    login_response = client.post(
        "/api/v1/auth/login",
        json={"email": test_user.email, "password": "newpassword123"},
    )
    assert login_response.status_code == 200


def test_change_password_wrong_current_password(client, test_user, auth_headers):
    """Test changing password with wrong current password."""
    response = client.post(
        "/api/v1/auth/change-password",
        json={
            "current_password": "wrongpassword",
            "new_password": "newpassword123",
        },
        headers=auth_headers,
    )

    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
