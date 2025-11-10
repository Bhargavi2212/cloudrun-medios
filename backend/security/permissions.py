"""Role and permission utilities."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List


class UserRole(str, Enum):
    ADMIN = "ADMIN"
    DOCTOR = "DOCTOR"
    NURSE = "NURSE"
    RECEPTIONIST = "RECEPTIONIST"
    GUEST = "GUEST"


ROLE_DEFAULT_PERMISSIONS: Dict[str, List[str]] = {
    UserRole.ADMIN.value: ["*"],
    UserRole.DOCTOR.value: [
        "consultation.view",
        "consultation.update",
        "notes.create",
        "notes.update",
        "summaries.view",
    ],
    UserRole.NURSE.value: [
        "consultation.view",
        "triage.submit_vitals",
        "queue.view",
    ],
    UserRole.RECEPTIONIST.value: [
        "queue.view",
        "checkin.create",
        "patient.create",
        "patient.view",
    ],
    UserRole.GUEST.value: [],
}


__all__ = ["UserRole", "ROLE_DEFAULT_PERMISSIONS"]
