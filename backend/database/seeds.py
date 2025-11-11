"""Database seed script for local development."""

from __future__ import annotations

from datetime import date

from faker import Faker

from backend.security.password import password_hasher

from .crud import create_instance, seed_roles
from .models import Patient, Role, User, UserRole, UserStatus
from .session import get_session

fake = Faker()

DEFAULT_ROLES = [
    {"name": "ADMIN", "description": "System administrator with full permissions."},
    {"name": "DOCTOR", "description": "Physician responsible for consultations."},
    {"name": "NURSE", "description": "Nurse handling vitals and triage."},
    {"name": "RECEPTIONIST", "description": "Reception staff handling check-ins."},
]

DEFAULT_USERS = [
    {
        "email": "admin@medios.ai",
        "password": "Password123!",
        "first_name": "System",
        "last_name": "Admin",
        "role_name": "ADMIN",
    },
    {
        "email": "doctor@medios.ai",
        "password": "Password123!",
        "first_name": "Dana",
        "last_name": "Doctor",
        "role_name": "DOCTOR",
    },
    {
        "email": "nurse@medios.ai",
        "password": "Password123!",
        "first_name": "Nina",
        "last_name": "Nurse",
        "role_name": "NURSE",
    },
    {
        "email": "receptionist@medios.ai",
        "password": "Password123!",
        "first_name": "Riley",
        "last_name": "Reception",
        "role_name": "RECEPTIONIST",
    },
]


def _ensure_user(
    session,
    *,
    email: str,
    password: str,
    first_name: str,
    last_name: str,
    role_name: str,
) -> None:
    existing = session.query(User).filter(User.email == email).filter(User.is_deleted.is_(False)).first()
    if existing:
        return

    role = session.query(Role).filter(Role.name == role_name, Role.is_deleted.is_(False)).first()
    if role is None:
        raise ValueError(f"Required role '{role_name}' not found. Seed roles before users.")

    user = create_instance(
        session,
        User,
        email=email,
        password_hash=password_hasher.hash(password),
        first_name=first_name,
        last_name=last_name,
        status=UserStatus.ACTIVE,
    )
    create_instance(
        session,
        UserRole,
        user_id=str(user.id),
        role_id=str(role.id),
    )


def seed_users() -> None:
    with get_session() as session:
        seed_roles(session, DEFAULT_ROLES)
        for user in DEFAULT_USERS:
            _ensure_user(session, **user)


def seed_patients(count: int = 50) -> None:
    with get_session() as session:
        seed_roles(session, DEFAULT_ROLES)
        for user in DEFAULT_USERS:
            _ensure_user(session, **user)

        existing = session.query(Patient).count()
        to_create = max(0, count - existing)
        for _ in range(to_create):
            create_instance(
                session,
                Patient,
                mrn=fake.unique.bothify(text="MRN-####-????"),
                first_name=fake.first_name(),
                last_name=fake.last_name(),
                date_of_birth=fake.date_between(date(1940, 1, 1), date(2010, 12, 31)),
                sex=fake.random_element(elements=("Male", "Female")),
                contact_phone=fake.phone_number(),
                contact_email=fake.email(),
            )


if __name__ == "__main__":
    seed_patients()
