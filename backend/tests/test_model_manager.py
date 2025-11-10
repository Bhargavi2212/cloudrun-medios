from pathlib import Path

import pytest

from backend.services import model_manager


@pytest.fixture(autouse=True)
def reset_singletons(monkeypatch):
    model_manager._WHISPER_MODEL = None
    model_manager._WHISPER_ERROR = None
    model_manager.get_settings.cache_clear()
    monkeypatch.setenv("WHISPER_CACHE_DIR", str(Path.cwd() / "tmp-models"))


def test_initialize_models_success(monkeypatch):
    load_calls = []

    def fake_load_model(name, download_root=None):
        load_calls.append((name, download_root))
        return object()

    monkeypatch.setattr(
        model_manager,
        "whisper",
        type("WhisperModule", (), {"load_model": staticmethod(fake_load_model)}),
    )
    monkeypatch.setenv("MODEL_SIZE", "tiny")

    success, error = model_manager.initialize_models()

    assert success is True
    assert error is None
    assert load_calls
    assert model_manager._WHISPER_MODEL is not None


def test_initialize_models_failure(monkeypatch):
    def raise_error(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        model_manager,
        "whisper",
        type("WhisperModule", (), {"load_model": staticmethod(raise_error)}),
    )

    success, error = model_manager.initialize_models()

    assert success is False
    assert error == "boom"
    assert model_manager._WHISPER_MODEL is None
