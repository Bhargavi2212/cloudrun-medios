from __future__ import annotations

import sys
import types

import pytest
from fastapi.testclient import TestClient

stub_whisper = types.ModuleType("whisper")
stub_whisper.load_model = lambda *args, **kwargs: None
sys.modules.setdefault("whisper", stub_whisper)

stub_google = types.ModuleType("google")
stub_generativeai = types.ModuleType("google.generativeai")
stub_generativeai.GenerativeModel = lambda *args, **kwargs: None
stub_generativeai.configure = lambda *args, **kwargs: None
sys.modules.setdefault("google", stub_google)
sys.modules.setdefault("google.generativeai", stub_generativeai)

from backend.main import app  # noqa: E402


def test_validation_error_returns_standard_response():
    client = TestClient(app)
    response = client.post("/api/v1/triage/predict", json={"features": "invalid"})
    data = response.json()

    assert response.status_code == 422
    assert data["success"] is False
    assert data["error"] == "Validation failed."
    assert "data" in data

