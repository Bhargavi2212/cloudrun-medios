import os
from pathlib import Path

from backend.services.config import get_settings
from backend.services.rate_limit import get_rate_limit_config
from backend.services.feature_flags import FeatureFlagService


def setup_function():
    get_settings.cache_clear()


def test_get_settings_creates_directories(tmp_path, monkeypatch):
    data_dir = tmp_path / "uploads"
    cache_dir = tmp_path / "models"
    storage_dir = tmp_path / "storage"
    monkeypatch.setenv("DATA_PATH", str(data_dir))
    monkeypatch.setenv("WHISPER_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("STORAGE_LOCAL_PATH", str(storage_dir))
    monkeypatch.setenv("MODEL_SIZE", "tiny")

    settings = get_settings()

    assert settings.data_path == data_dir
    assert settings.whisper_cache_dir == cache_dir
    assert settings.storage_local_path == storage_dir
    assert data_dir.exists()
    assert cache_dir.exists()
    assert storage_dir.exists()


def test_get_settings_invalid_model_size_defaults(monkeypatch):
    monkeypatch.delenv("DATA_PATH", raising=False)
    monkeypatch.delenv("WHISPER_CACHE_DIR", raising=False)
    monkeypatch.setenv("MODEL_SIZE", "invalid-size")
    settings = get_settings()
    assert settings.model_size == "base"


def test_feature_flags_and_rate_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "uploads"))
    monkeypatch.setenv("WHISPER_CACHE_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("STORAGE_LOCAL_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv(
        "FEATURE_FLAGS",
        '{"beta_dashboard": true, "new_notes": false, "analytics": true}',
    )
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("RATE_LIMIT_DEFAULT_PER_MINUTE", "200")
    monkeypatch.setenv("RATE_LIMIT_BURST_MULTIPLIER", "2.5")

    settings = get_settings()
    flags = FeatureFlagService(settings.feature_flags)
    rl = get_rate_limit_config()

    assert flags.is_enabled("beta_dashboard") is True
    assert flags.is_enabled("new_notes") is False
    assert flags.is_enabled("analytics") is True
    assert rl.enabled is True
    assert rl.default_per_minute == 200
    assert rl.burst_limit == int(200 * 2.5)

