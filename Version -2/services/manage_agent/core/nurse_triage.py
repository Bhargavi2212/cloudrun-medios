"""
Nurse triage model that uses vitals + receptionist signals.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.parent.parent.parent
MODEL_DIR = ROOT_DIR / "data" / "nurse_models" / "models"
TRANSFORMER_DIR = ROOT_DIR / "data" / "processed" / "transformers"
PREPROCESSING_PARAMS_PATH = (
    ROOT_DIR / "data" / "processed" / "preprocessing_params.json"
)

RFV_CLUSTERS = [
    "rfv1_cluster_Ear_Nose_Throat",
    "rfv1_cluster_Fever_Infection",
    "rfv1_cluster_Gastrointestinal",
    "rfv1_cluster_General_Symptoms",
    "rfv1_cluster_Mental_Health",
    "rfv1_cluster_Musculoskeletal",
    "rfv1_cluster_Neurological",
    "rfv1_cluster_Other",
    "rfv1_cluster_Respiratory",
    "rfv1_cluster_Skin",
    "rfv1_cluster_Trauma_Injury",
    "rfv1_cluster_Urinary_Genitourinary",
]

FEATURE_ORDER = [
    "pulse_yj",
    "respiration_yj",
    "sbp",
    "dbp",
    "temp_c_yj",
    "pain",
    "age",
    "ambulance_arrival",
    "seen_72h",
    "injury",
    "pain_missing",
    *RFV_CLUSTERS,
]

DEFAULT_IMPUTATIONS = {
    "pulse": 88.0,
    "respiration": 18.0,
    "temp_c": 36.8,
    "sbp": 132.0,
    "dbp": 78.0,
    "pain": 5.0,
    "age": 36.0,
}


@dataclass
class NurseTriagePayload:
    """Payload containing vitals and contextual information."""

    hr: float | None
    rr: float | None
    sbp: float | None
    dbp: float | None
    temp_c: float | None
    pain: float | None
    age: float | None
    chief_complaint: str | None
    ambulance_arrival: bool = False
    seen_72h: bool = False
    injury: bool = False


@dataclass
class NurseTriageResult:
    """Result returned by the nurse triage model."""

    acuity_level: int
    model_version: str
    explanation: str
    probabilities: dict[str, float] | None


class NurseTriageEngine:
    """
    Executes the trained nurse RandomForest model using vitals + receptionist signals.
    """

    def __init__(
        self,
        *,
        model_path: Path | None = None,
        model_version: str = "nurse-rf-v1",
    ) -> None:
        self.model_version = model_version
        self.model_path = model_path or (MODEL_DIR / "random_forest.pkl")
        self.model: BaseEstimator | None = None
        self.power_transformer = None
        self.robust_scaler = None
        self.imputation_values: dict[str, float] = {}

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load model and preprocessing artifacts."""
        try:
            self.model = joblib.load(self.model_path)
            logger.info("Loaded nurse triage model from %s", self.model_path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load nurse triage model: %s", exc)
            self.model = None

        try:
            self.power_transformer = joblib.load(
                TRANSFORMER_DIR / "power_transformer.pkl"
            )
            self.robust_scaler = joblib.load(TRANSFORMER_DIR / "robust_scaler.pkl")
            logger.info("Loaded transformer + scaler for nurse triage")
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load nurse preprocessing transformers: %s", exc)
            self.power_transformer = None
            self.robust_scaler = None

        if PREPROCESSING_PARAMS_PATH.exists():
            with PREPROCESSING_PARAMS_PATH.open("r", encoding="utf-8") as handle:
                params: dict[str, Any] = json.load(handle)
            self.imputation_values = params.get("imputation_values", {})
        else:  # pragma: no cover - fallback
            logger.warning(
                "Preprocessing params not found at %s", PREPROCESSING_PARAMS_PATH
            )

    def _value_or_median(self, key: str, value: float | None) -> float:
        if value is None:
            return float(
                self.imputation_values.get(
                    f"{key}_median", DEFAULT_IMPUTATIONS.get(key, 0.0)
                )
            )
        return float(value)

    def _map_chief_complaint(self, chief_complaint: str | None) -> dict[str, int]:
        clusters = {cluster: 0 for cluster in RFV_CLUSTERS}
        if not chief_complaint:
            clusters["rfv1_cluster_Other"] = 1
            return clusters

        text = chief_complaint.lower()

        keyword_map = {
            "rfv1_cluster_Ear_Nose_Throat": [
                "ear",
                "throat",
                "sinus",
                "nose",
                "tonsil",
            ],
            "rfv1_cluster_Fever_Infection": ["fever", "infection", "flu", "cold"],
            "rfv1_cluster_Gastrointestinal": [
                "stomach",
                "abdomen",
                "vomit",
                "diarrhea",
                "nausea",
                "gastric",
            ],
            "rfv1_cluster_General_Symptoms": [
                "tired",
                "weakness",
                "fatigue",
                "malaise",
            ],
            "rfv1_cluster_Mental_Health": ["anxiety", "depression", "mental", "psych"],
            "rfv1_cluster_Musculoskeletal": [
                "back",
                "knee",
                "joint",
                "muscle",
                "sprain",
                "strain",
            ],
            "rfv1_cluster_Neurological": [
                "headache",
                "migraine",
                "seizure",
                "stroke",
                "dizzy",
            ],
            "rfv1_cluster_Respiratory": [
                "cough",
                "breath",
                "asthma",
                "respiratory",
                "shortness of breath",
            ],
            "rfv1_cluster_Skin": ["rash", "skin", "burn", "itch"],
            "rfv1_cluster_Trauma_Injury": [
                "trauma",
                "injury",
                "accident",
                "fall",
                "fracture",
                "laceration",
                "wound",
            ],
            "rfv1_cluster_Urinary_Genitourinary": [
                "urinary",
                "uti",
                "kidney",
                "pelvic",
                "genital",
            ],
        }

        for cluster, keywords in keyword_map.items():
            if any(keyword in text for keyword in keywords):
                clusters[cluster] = 1
                return clusters

        clusters["rfv1_cluster_Other"] = 1
        return clusters

    def _prepare_features(self, payload: NurseTriagePayload) -> dict[str, float]:
        if self.power_transformer is None or self.robust_scaler is None:
            raise RuntimeError("Nurse triage preprocessors are not loaded.")

        pulse = self._value_or_median("pulse", payload.hr)
        respiration = self._value_or_median("respiration", payload.rr)
        temp_c = self._value_or_median("temp_c", payload.temp_c)
        sbp = self._value_or_median("sbp", payload.sbp)
        dbp = self._value_or_median("dbp", payload.dbp)
        pain_missing = 1 if payload.pain is None else 0
        pain = self._value_or_median(
            "pain", payload.pain if payload.pain is not None else None
        )
        age = self._value_or_median("age", payload.age)

        transformed = self.power_transformer.transform([[pulse, respiration, temp_c]])[
            0
        ]
        continuous_vector = [
            float(transformed[0]),
            float(transformed[1]),
            sbp,
            dbp,
            float(transformed[2]),
            pain,
            age,
        ]
        scaled = self.robust_scaler.transform([continuous_vector])[0]

        features: dict[str, float] = {
            "pulse_yj": scaled[0],
            "respiration_yj": scaled[1],
            "sbp": scaled[2],
            "dbp": scaled[3],
            "temp_c_yj": scaled[4],
            "pain": scaled[5],
            "age": scaled[6],
            "ambulance_arrival": 1.0 if payload.ambulance_arrival else 0.0,
            "seen_72h": 1.0 if payload.seen_72h else 0.0,
            "injury": 1.0 if payload.injury else 0.0,
            "pain_missing": float(pain_missing),
        }

        features.update(self._map_chief_complaint(payload.chief_complaint))

        return features

    def classify(self, payload: NurseTriagePayload) -> NurseTriageResult:
        if self.model is None:
            raise RuntimeError("Nurse triage model is not loaded.")

        features = self._prepare_features(payload)
        feature_vector = np.array([[features[name] for name in FEATURE_ORDER]])
        prediction = self.model.predict(feature_vector)[0]
        acuity = int(prediction)

        probabilities: dict[str, float] | None = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(feature_vector)[0]
            probabilities = {
                str(index): float(value) for index, value in enumerate(proba, start=1)
            }

        explanation = (
            f"Nurse RandomForest prediction (model {self.model_version}) "
            f"using vitals and receptionist signals."
        )

        return NurseTriageResult(
            acuity_level=max(1, min(5, acuity)),
            model_version=self.model_version,
            explanation=explanation,
            probabilities=probabilities,
        )


__all__ = ["NurseTriageEngine", "NurseTriagePayload", "NurseTriageResult"]
