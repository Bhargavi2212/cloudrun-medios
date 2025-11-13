from __future__ import annotations

from typing import Iterable, List, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database.models import Role, User, UserRole, UserStatus
from backend.database.session import get_session
from backend.security.password import password_hasher
from backend.services.logging import get_logger

logger = get_logger(__name__)

DEFAULT_ROLES: Tuple[dict[str, str], ...] = (
    {"name": "ADMIN", "description": "System administrator with full permissions."},
    {"name": "DOCTOR", "description": "Physician responsible for consultations."},
    {"name": "NURSE", "description": "Nurse handling vitals and triage."},
    {"name": "RECEPTIONIST", "description": "Reception staff handling check-ins."},
)

DEMO_USERS: Tuple[dict[str, str], ...] = (
    {
        "email": "admin@medios.ai",
        "password": "Password123!",
        "first_name": "System",
        "last_name": "Admin",
        "role": "ADMIN",
    },
    {
        "email": "doctor@medios.ai",
        "password": "Password123!",
        "first_name": "Dana",
        "last_name": "Doctor",
        "role": "DOCTOR",
    },
    {
        "email": "nurse@medios.ai",
        "password": "Password123!",
        "first_name": "Nina",
        "last_name": "Nurse",
        "role": "NURSE",
    },
    {
        "email": "receptionist@medios.ai",
        "password": "Password123!",
        "first_name": "Riley",
        "last_name": "Reception",
        "role": "RECEPTIONIST",
    },
)


def _get_role(session: Session, role_name: str) -> Role | None:
    stmt = select(Role).where(Role.name == role_name).where(Role.is_deleted.is_(False)).limit(1)
    return session.execute(stmt).scalar_one_or_none()


def _get_user(session: Session, email: str) -> User | None:
    stmt = select(User).where(User.email == email.lower()).where(User.is_deleted.is_(False)).limit(1)
    return session.execute(stmt).scalar_one_or_none()


def _ensure_roles(session: Session) -> List[str]:
    created: List[str] = []
    for role_data in DEFAULT_ROLES:
        role = _get_role(session, role_data["name"])
        if role:
            continue
        role = Role(name=role_data["name"], description=role_data["description"])
        session.add(role)
        created.append(role.name)
    if created:
        logger.info("Roles seeded: %s", ", ".join(created))
    else:
        logger.info("Roles already initialized")
    return created


def _create_demo_user(session: Session, user_data: dict[str, str]) -> None:
    role = _get_role(session, user_data["role"])
    if role is None:
        raise RuntimeError(f"Required role '{user_data['role']}' not found. Seed roles before users.")

    user = User(
        email=user_data["email"].lower(),
        password_hash=password_hasher.hash(user_data["password"]),
        first_name=user_data["first_name"],
        last_name=user_data["last_name"],
        status=UserStatus.ACTIVE,
    )
    session.add(user)
    session.flush()  # ensure user.id is available
    session.add(UserRole(user_id=str(user.id), role_id=str(role.id)))


def _ensure_demo_users(session: Session) -> Iterable[str]:
    total_users = session.execute(select(User.id).where(User.is_deleted.is_(False)).limit(1)).first()

    if total_users is not None:
        logger.info("Existing users detected - skipping demo user creation")
        return []

    created: List[str] = []
    for user_data in DEMO_USERS:
        _create_demo_user(session, user_data)
        created.append(user_data["email"])

    logger.info("Demo users seeded: %s", ", ".join(created))
    return created


def _ensure_role_assignments(session: Session) -> List[str]:
    assigned: List[str] = []
    for user_data in DEMO_USERS:
        user = _get_user(session, user_data["email"])
        if user is None:
            continue

        target_role = _get_role(session, user_data["role"])
        if target_role is None:
            logger.warning("Role '%s' missing while assigning to %s", user_data["role"], user.email)
            continue

        if any(role.id == target_role.id for role in user.roles):
            continue

        user.roles.append(target_role)
        assigned.append(user.email)

    if assigned:
        logger.info("Assigned missing roles for demo users: %s", ", ".join(assigned))
    return assigned


def initialize_demo_data() -> None:
    """Ensure required roles and demo users exist."""
    with get_session() as session:
        _ensure_roles(session)
        created_users = list(_ensure_demo_users(session))
        if not created_users:
            _ensure_role_assignments(session)
