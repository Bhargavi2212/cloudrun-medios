from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from backend.services import triage_service as triage_module
from backend.services.triage_service import TriageService


class DummyXGBBooster:
    def predict(self, dmatrix, pred_contribs: bool = True):
        # Return contributions for 4 features + bias
        return np.array([[0.5, -0.2, 0.1, 0.05, 0.0]])


class DummyXGBModel:
    def predict_proba(self, X):
        return np.array([[0.05, 0.10, 0.15, 0.30, 0.40]])

    def get_booster(self):
        booster = DummyXGBBooster()
        booster.feature_names = ["f0", "f1", "f2", "f3"]
        return booster


class DummyLGBBooster:
    def predict(self, X, pred_contrib: bool = False):
        return np.array([[0.3, -0.1, 0.2, 0.05, 0.0]])


class DummyLGBModel:
    def __init__(self) -> None:
        self.booster_ = DummyLGBBooster()

    def predict_proba(self, X):
        return np.array([[0.10, 0.15, 0.20, 0.25, 0.30]])


class DummyStackingModel:
    def predict_proba(self, X):
        return np.array([[0.12, 0.18, 0.22, 0.24, 0.24]])


class DummySettings:
    triage_model_dir = Path(".")
    triage_metadata_file = Path("meta.pkl")
    triage_baseline_metadata_file = Path("baseline.pkl")
    triage_xgb_model = Path("xgb.pkl")
    triage_lgbm_model = Path("lgbm.pkl")
    triage_stacking_model = Path("stack.pkl")


@pytest.fixture(autouse=True)
def patch_triage_dependencies(monkeypatch):
    monkeypatch.setattr(triage_module, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(triage_module, "record_service_metric", lambda **kwargs: None)

    def fake_resolve(self, path_like):
        return Path(".")

    def fake_pickle(self, filename):
        if "baseline" in str(filename):
            return {"feature_names": ["f0", "f1", "f2", "f3"]}
        return {
            "classes": 5,
            "severity_labels_0based": {
                0: "Critical",
                1: "Emergent",
                2: "Urgent",
                3: "Standard",
                4: "Non-urgent",
            },
        }

    def fake_joblib(self, filename):
        name = Path(filename).stem
        if name == "xgb":
            return DummyXGBModel()
        if name == "lgbm":
            return DummyLGBModel()
        return DummyStackingModel()

    monkeypatch.setattr(TriageService, "_resolve_path", fake_resolve, raising=False)
    monkeypatch.setattr(TriageService, "_load_pickle", fake_pickle, raising=False)
    monkeypatch.setattr(TriageService, "_load_joblib", fake_joblib, raising=False)


def test_triage_service_default_model_is_lightgbm():
    service = TriageService()

    assert service.default_model == "lightgbm"
    assert sorted(service.severity_labels.keys()) == [0, 1, 2, 3, 4]
    assert service.feature_names == ["f0", "f1", "f2", "f3"]


def test_predict_returns_lightgbm_probabilities_and_explanations():
    service = TriageService()
    features = {name: 0 for name in service.feature_names}

    result = service.predict(features, top_k=2)

    assert result.model_used == "lightgbm"
    assert pytest.approx(sum(result.probabilities.values()), 0.001) == 1.0
    assert len(result.explanation) == 2
    assert result.latency_ms >= 0.0


def test_predict_allows_model_override_to_xgboost():
    service = TriageService()
    features = {name: 0 for name in service.feature_names}

    result = service.predict(features, top_k=3, model="xgboost")

    assert result.model_used == "xgboost"
    assert result.severity_label in service.severity_labels.values()
    assert len(result.explanation) == 3


def test_predict_rejects_unsupported_model():
    service = TriageService()
    features = {name: 0 for name in service.feature_names}

    with pytest.raises(ValueError):
        service.predict(features, model="invalid-model")
