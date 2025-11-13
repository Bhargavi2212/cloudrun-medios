from __future__ import annotations

import pytest

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from backend.security.permissions import UserRole


def _combine_path_method(path_method: tuple[str, str]) -> str:
    path, method = path_method
    return f"{method.upper()} {path}"


@pytest.fixture
def make_headers(token_factory):
    def _make_headers(role: UserRole) -> dict[str, str]:
        access_token = token_factory(role=role.value)
        return {"Authorization": f"Bearer {access_token}"}

    return _make_headers


@pytest.fixture
def role_allowed_roles():
    return {
        "GET /api/v1/patients": {
            UserRole.RECEPTIONIST,
            UserRole.NURSE,
            UserRole.DOCTOR,
            UserRole.ADMIN,
        },
        "POST /api/v1/patients": {
            UserRole.RECEPTIONIST,
            UserRole.NURSE,
            UserRole.ADMIN,
        },
        "POST /api/v1/triage/predict": {
            UserRole.NURSE,
            UserRole.DOCTOR,
            UserRole.ADMIN,
        },
        "GET /api/v1/summarizer/{subject_id}": {
            UserRole.DOCTOR,
            UserRole.ADMIN,
        },
        "POST /api/v1/queue": {
            UserRole.RECEPTIONIST,
            UserRole.NURSE,
            UserRole.ADMIN,
        },
        "POST /api/v1/queue/{queue_state_id}/advance": {
            UserRole.NURSE,
            UserRole.DOCTOR,
            UserRole.ADMIN,
        },
    }


@pytest.mark.parametrize(
    "endpoint",
    [
        ("/api/v1/patients", "get"),
        ("/api/v1/patients", "post"),
    ],
)
@pytest.mark.parametrize(
    "role",
    [UserRole.RECEPTIONIST, UserRole.NURSE, UserRole.DOCTOR, UserRole.ADMIN],
)
def test_patients_access(client, make_headers, endpoint, role, role_allowed_roles, monkeypatch):
    path, method = endpoint
    allowed_roles = role_allowed_roles.get(_combine_path_method(endpoint), set())
    expected_status = 403
    if role in allowed_roles:
        expected_status = 201 if method == "post" else 200

    payload = {}
    if method == "post":
        payload = {
            "first_name": "Test",
            "last_name": "User",
            "sex": "Other",
            "contact_email": "test@example.com",
        }
        monkeypatch.setattr("backend.database.crud.get_patient_by_mrn", lambda session, mrn: None)

        def create_patient_stub(session, **kwargs):
            return SimpleNamespace(
                id=uuid4(),
                mrn=kwargs.get("mrn", "MRN-123"),
                first_name=kwargs.get("first_name", "Test"),
                last_name=kwargs.get("last_name", "User"),
                date_of_birth=kwargs.get("date_of_birth"),
                sex=kwargs.get("sex", "Other"),
                contact_phone=kwargs.get("contact_phone"),
                contact_email=kwargs.get("contact_email"),
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

        monkeypatch.setattr("backend.database.crud.create_patient", create_patient_stub)
    else:
        sample_patient = SimpleNamespace(
            id=uuid4(),
            mrn="MRN-456",
            first_name="Sample",
            last_name="Patient",
            date_of_birth=None,
            sex="Other",
            contact_phone=None,
            contact_email="sample@example.com",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        monkeypatch.setattr(
            "backend.database.crud.list_patients",
            lambda session, skip, limit: [sample_patient],
        )
        monkeypatch.setattr("backend.database.crud.count_patients", lambda session: 0)

    request_kwargs = {"headers": make_headers(role)}
    if method == "post" and payload:
        request_kwargs["json"] = payload

    response = getattr(client, method)(path, **request_kwargs)
    assert response.status_code == expected_status


@pytest.mark.parametrize("role", [UserRole.NURSE, UserRole.DOCTOR, UserRole.ADMIN, UserRole.RECEPTIONIST])
def test_triage_predict_access(client, make_headers, role, role_allowed_roles, monkeypatch):
    allowed_roles = role_allowed_roles.get("POST /api/v1/triage/predict", set())
    expected_status = 200 if role in allowed_roles else 403

    def predict_stub(features, top_k=5, model=None, use_shap=False):
        return SimpleNamespace(
            severity_index=3,
            severity_label="Moderate",
            probabilities=[0.1, 0.2, 0.4, 0.2, 0.1],
            explanation={"feature": "importance"},
            model_used="lightgbm",
            latency_ms=12.3,
        )

    monkeypatch.setattr("backend.api.v1.triage.service.predict", predict_stub)
    response = client.post(
        "/api/v1/triage/predict",
        headers=make_headers(role),
        json={"features": {"heart_rate": 80, "blood_pressure_systolic": 120}},
    )
    assert response.status_code == expected_status


@pytest.mark.parametrize("role", [UserRole.DOCTOR, UserRole.ADMIN, UserRole.NURSE])
def test_summarizer_access(client, make_headers, role, role_allowed_roles, monkeypatch):
    allowed_roles = role_allowed_roles.get("GET /api/v1/summarizer/{subject_id}", set())
    expected_status = 200 if role in allowed_roles else 403

    def summarize_stub(self, subject_id, visit_limit=None, force_refresh=False):
        return SimpleNamespace(
            subject_id=subject_id,
            summary_markdown="Summary",
            timeline={},
            metrics={},
            cached=False,
            is_stub=False,
        )

    monkeypatch.setattr(
        "backend.services.summarizer_service.MedicalSummarizer.summarize_patient",
        summarize_stub,
    )
    response = client.get("/api/v1/summarizer/1", headers=make_headers(role))
    assert response.status_code == expected_status


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/api/v1/queue"),
        ("post", "/api/v1/queue/{queue_state_id}/advance"),
    ],
)
@pytest.mark.parametrize("role", [UserRole.RECEPTIONIST, UserRole.NURSE, UserRole.ADMIN, UserRole.DOCTOR])
def test_queue_access(client, make_headers, method, path, role, role_allowed_roles, monkeypatch):
    allowed_roles = role_allowed_roles.get(f"{method.upper()} {path}", set())
    expected_status = 403
    if role in allowed_roles:
        expected_status = 201 if path == "/api/v1/queue" else 200

    if path == "/api/v1/queue":

        def create_entry_stub(*args, **kwargs):
            return SimpleNamespace(model_dump=lambda: {"id": "queue-entry"})

        monkeypatch.setattr(
            "backend.services.queue_service.queue_service.create_entry",
            create_entry_stub,
        )

    if path == "/api/v1/queue/{queue_state_id}/advance":

        def transition_stub(*args, **kwargs):
            return SimpleNamespace(model_dump=lambda: {"id": "queue-advance"})

        monkeypatch.setattr(
            "backend.services.queue_service.queue_service.transition_stage",
            transition_stub,
        )

    data = {
        "/api/v1/queue": {
            "patient_id": "patient",
            "chief_complaint": "complaint",
        },
        "/api/v1/queue/{queue_state_id}/advance": {"next_stage": "triage"},
    }

    url = path.replace("{queue_state_id}", "abc123")
    response = getattr(client, method)(
        url,
        headers=make_headers(role),
        json=data.get(path),
    )
    assert response.status_code == expected_status

