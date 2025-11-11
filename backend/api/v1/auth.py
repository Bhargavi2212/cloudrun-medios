from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

from backend.services.auth_service import AuthService
from backend.services.error_response import StandardResponse

auth_service = AuthService()


class UserResponse(BaseModel):
    id: UUID
    email: EmailStr
    first_name: Optional[str]
    last_name: Optional[str]
    phone: Optional[str]
    status: str
    last_login_at: Optional[str]
    created_at: str
    updated_at: str
    full_name: str
    role: str
    roles: List[str]

    @classmethod
    def from_dict(cls, data: dict) -> "UserResponse":
        return cls(
            id=UUID(str(data["id"])),
            email=data["email"],
            first_name=data.get("first_name"),
            last_name=data.get("last_name"),
            phone=data.get("phone"),
            status=str(data.get("status")),
            last_login_at=data.get("last_login_at"),
            created_at=str(data.get("created_at")),
            updated_at=str(data.get("updated_at")),
            full_name=data.get("full_name", ""),
            role=data.get("role", "GUEST"),
            roles=data.get("roles", []),
        )


from backend.security.dependencies import get_current_user
from backend.services.auth_service import AuthService
from backend.services.error_response import StandardResponse

router = APIRouter(prefix="/auth", tags=["auth"])
auth_service = AuthService()


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    first_name: str | None = None
    last_name: str | None = None
    roles: List[str]


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserResponse


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str


class UpdateProfileRequest(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=8)
    new_password: str = Field(min_length=8)


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    reset_token: str
    new_password: str = Field(min_length=8)


def _build_user_response(user) -> UserResponse:
    payload = auth_service.serialize_user(user)
    return UserResponse.from_dict(payload)


@router.post("/register", response_model=StandardResponse, status_code=status.HTTP_201_CREATED)
def register(request: RegisterRequest) -> StandardResponse:
    user = auth_service.register_user(
        email=request.email,
        password=request.password,
        first_name=request.first_name,
        last_name=request.last_name,
        role_names=request.roles,
    )
    return StandardResponse(success=True, data=_build_user_response(user))


@router.post("/login", response_model=StandardResponse)
def login(request: LoginRequest) -> StandardResponse:
    access_token, refresh_token, user = auth_service.authenticate_user(request.email, request.password)
    token_response = TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=_build_user_response(user),
    )
    return StandardResponse(success=True, data=token_response)


@router.post("/refresh", response_model=StandardResponse)
def refresh_tokens(request: RefreshRequest) -> StandardResponse:
    access_token, refresh_token = auth_service.refresh_tokens(request.refresh_token)
    return StandardResponse(
        success=True,
        data={
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
        },
    )


@router.post("/logout", response_model=StandardResponse, status_code=status.HTTP_200_OK)
def logout(request: LogoutRequest) -> StandardResponse:
    auth_service.revoke_refresh_token(request.refresh_token)
    return StandardResponse(success=True)


@router.get("/me", response_model=StandardResponse)
def me(current_user=Depends(get_current_user)) -> StandardResponse:
    return StandardResponse(success=True, data=_build_user_response(current_user))


@router.put("/me", response_model=StandardResponse)
def update_profile(request: UpdateProfileRequest, current_user=Depends(get_current_user)) -> StandardResponse:
    user = auth_service.update_user_profile(
        str(current_user.id),
        first_name=request.first_name,
        last_name=request.last_name,
        phone=request.phone,
    )
    return StandardResponse(success=True, data=_build_user_response(user))


@router.post("/change-password", response_model=StandardResponse)
def change_password(request: ChangePasswordRequest, current_user=Depends(get_current_user)) -> StandardResponse:
    auth_service.change_password(str(current_user.id), request.current_password, request.new_password)
    return StandardResponse(success=True)


@router.post("/forgot-password", response_model=StandardResponse)
def forgot_password(request: ForgotPasswordRequest) -> StandardResponse:
    """Request a password reset token.

    For security, this always returns success even if the email doesn't exist.
    In production, the reset token would be sent via email.
    """
    reset_token = auth_service.request_password_reset(request.email)
    # For development/testing, include the token in the response
    # In production, this would be None and the token would be sent via email
    return StandardResponse(
        success=True,
        data={"reset_token": reset_token} if reset_token else None,
        message=(
            "If the email exists, a password reset link has been sent."
            if not reset_token
            else "Password reset token generated."
        ),
    )


@router.post("/reset-password", response_model=StandardResponse)
def reset_password(request: ResetPasswordRequest) -> StandardResponse:
    """Reset password using a reset token."""
    auth_service.reset_password(request.reset_token, request.new_password)
    return StandardResponse(success=True, message="Password has been reset successfully.")
