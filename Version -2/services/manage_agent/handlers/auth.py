"""
Simple authentication endpoint for demo users.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from shared.schemas import StandardResponse

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    """Login request payload."""

    email: str = Field(..., description="User email address.")
    password: str = Field(..., description="User password.")


class UserResponse(BaseModel):
    """User response model."""

    id: str = Field(..., description="User identifier.")
    email: str = Field(..., description="User email.")
    first_name: str | None = Field(None, description="First name.")
    last_name: str | None = Field(None, description="Last name.")
    full_name: str = Field(..., description="Full name.")
    role: str = Field(..., description="User role.")
    roles: list[str] = Field(default_factory=list, description="User roles.")


class TokenResponse(BaseModel):
    """Token response model."""

    access_token: str = Field(..., description="JWT access token.")
    refresh_token: str = Field(..., description="Refresh token.")
    user: UserResponse = Field(..., description="User information.")


DEMO_PASSWORD = "demo123"

_BASE_DEMO_PROFILES = [
    {
        "id": "demo-receptionist",
        "first_name": "Jane",
        "last_name": "Receptionist",
        "full_name": "Jane Receptionist",
        "role": "RECEPTIONIST",
        "emails": ["receptionist@demo.com", "receptionist@hospital.com"],
    },
    {
        "id": "demo-nurse",
        "first_name": "John",
        "last_name": "Nurse",
        "full_name": "John Nurse",
        "role": "NURSE",
        "emails": ["nurse@demo.com", "nurse@hospital.com"],
    },
    {
        "id": "demo-doctor",
        "first_name": "Dr. Sarah",
        "last_name": "Doctor",
        "full_name": "Dr. Sarah Doctor",
        "role": "DOCTOR",
        "emails": ["doctor@demo.com", "doctor@hospital.com"],
    },
    {
        "id": "demo-admin",
        "first_name": "Admin",
        "last_name": "User",
        "full_name": "Admin User",
        "role": "ADMIN",
        "emails": ["admin@demo.com", "admin@hospital.com"],
    },
]

DEMO_USERS: dict[str, dict[str, str]] = {}
for profile in _BASE_DEMO_PROFILES:
    for email in profile["emails"]:
        DEMO_USERS[email] = {
            "id": profile["id"],
            "email": email,
            "first_name": profile["first_name"],
            "last_name": profile["last_name"],
            "full_name": profile["full_name"],
            "role": profile["role"],
            "password": DEMO_PASSWORD,
        }


@router.post(
    "/login",
    response_model=StandardResponse,
    summary="Login user",
    description="Authenticate user and return access token. Supports demo users.",
)
async def login(payload: LoginRequest) -> StandardResponse:
    """
    Authenticate user with email and password.
    For demo purposes, accepts predefined demo users.
    """

    email = payload.email.lower().strip()
    user_data = DEMO_USERS.get(email)

    if not user_data or user_data["password"] != payload.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    user = UserResponse(
        id=user_data["id"],
        email=user_data["email"],
        first_name=user_data["first_name"],
        last_name=user_data["last_name"],
        full_name=user_data["full_name"],
        role=user_data["role"],
        roles=[user_data["role"]],
    )

    # Generate simple tokens (in production, use proper JWT)
    access_token = f"demo_token_{user_data['id']}"
    refresh_token = f"demo_refresh_{user_data['id']}"

    token_response = TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=user,
    )

    return StandardResponse(success=True, data=token_response.model_dump())


@router.get(
    "/me",
    response_model=StandardResponse,
    summary="Get current user",
    description="Return the current authenticated user.",
)
async def get_current_user() -> StandardResponse:
    """
    Get current user (demo implementation).
    In production, this would extract user from JWT token.
    """

    # For demo, return a default user
    # In production, extract from Authorization header
    user = UserResponse(
        id="demo-receptionist",
        email="receptionist@demo.com",
        first_name="Jane",
        last_name="Receptionist",
        full_name="Jane Receptionist",
        role="RECEPTIONIST",
        roles=["RECEPTIONIST"],
    )

    return StandardResponse(success=True, data=user.model_dump())
