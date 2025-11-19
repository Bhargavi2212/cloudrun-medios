from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

from backend.database import crud
from backend.database.models import Patient
from backend.database.session import get_session
from backend.services.config import get_settings

logger = logging.getLogger(__name__)

NURSE_FEATURES = [
    "pulse_yj",
    "respiration_yj",
    "sbp",
    "dbp",
    "temp_c_yj",
    "pain",
    "pain_missing",
    "age",
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
    "ambulance_arrival",
    "seen_72h",
    "injury",
]


class NurseTriageBridge:
    """Loads the nurse RandomForest model and exposes prediction helpers."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        model_path = Path(self.settings.nurse_triage_model_path)
        if not model_path.exists():
            logger.warning("Nurse triage model not found at %s", model_path)
            return
        try:
            self.model = joblib.load(model_path)
            logger.info("Loaded nurse triage model from %s", model_path)
        except Exception as exc:
            logger.error("Failed to load nurse triage model: %s", exc)
            self.model = None

    def predict_for_session(self, session_id: str) -> Optional[dict]:
        if self.model is None:
            logger.warning("Nurse triage model unavailable; skipping prediction.")
            return None

        features_df = self._build_feature_frame(session_id)
        if features_df is None:
            return None

        start = datetime.now(timezone.utc)
        probs = self.model.predict_proba(features_df)[0]
        latency_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        esi_idx = int(np.argmax(probs))
        esi_level = esi_idx + 1
        probabilities = {str(idx + 1): float(value) for idx, value in enumerate(probs)}

        with get_session() as db:
            session_obj = crud.get_scribe_session(db, session_id)
            if session_obj is None:
                raise ValueError(f"Scribe session {session_id} not found.")
            if not session_obj.consultation_id:
                raise ValueError("Scribe session has no consultation_id; cannot record triage prediction.")
            prediction = crud.create_triage_prediction(
                db,
                session_id=session_id,
                consultation_id=session_obj.consultation_id,
                model_version="nurse_rf_v1",
                esi_level=esi_level,
                probability=float(probs[esi_idx]),
                probabilities=probabilities,
                flagged=esi_level in (1, 2),
                latency_ms=latency_ms,
                source="nurse_random_forest",
                inputs_snapshot=features_df.to_dict(orient="records")[0],
            )

        return {
            "prediction_id": prediction.id,
            "esi_level": esi_level,
            "probabilities": probabilities,
            "flagged": esi_level in (1, 2),
            "latency_ms": latency_ms,
        }

    def _build_feature_frame(self, session_id: str) -> Optional[pd.DataFrame]:
        with get_session() as db:
            session_obj = crud.get_scribe_session(db, session_id)
            if session_obj is None:
                return None
            vitals = crud.list_scribe_vitals(db, session_id, limit=1)
            latest_vitals = vitals[-1] if vitals else None
            patient: Optional[Patient] = None
            if session_obj.patient_id:
                patient = db.get(Patient, session_obj.patient_id)

        feature_values: Dict[str, float] = {feature: 0.0 for feature in NURSE_FEATURES}
        if latest_vitals:
            feature_values["pulse_yj"] = float(latest_vitals.heart_rate or 0.0)
            feature_values["respiration_yj"] = float(latest_vitals.respiratory_rate or 0.0)
            feature_values["sbp"] = float(latest_vitals.systolic_bp or 0.0)
            feature_values["dbp"] = float(latest_vitals.diastolic_bp or 0.0)
            feature_values["temp_c_yj"] = float(latest_vitals.temperature_c or 0.0)
            feature_values["pain"] = float(latest_vitals.pain_score or 0.0)
            feature_values["pain_missing"] = 0.0 if latest_vitals.pain_score is not None else 1.0
        else:
            feature_values["pain_missing"] = 1.0

        if patient and patient.date_of_birth:
            dob = patient.date_of_birth
            if isinstance(dob, datetime):
                dob = dob.date()
            today = datetime.now(timezone.utc).date()
            age_years = (today - dob).days // 365
            feature_values["age"] = float(max(age_years, 0))

        # Additional binary signals via session metadata (if present)
        metadata = getattr(session_obj, "session_metadata", {}) or {}
        for cluster_key in [key for key in NURSE_FEATURES if key.startswith("rfv1_cluster")]:
            feature_values[cluster_key] = 1.0 if metadata.get(cluster_key) else 0.0
        feature_values["ambulance_arrival"] = 1.0 if metadata.get("ambulance_arrival") else 0.0
        feature_values["seen_72h"] = 1.0 if metadata.get("seen_72h") else 0.0
        feature_values["injury"] = 1.0 if metadata.get("injury") else 0.0

        df = pd.DataFrame([feature_values], columns=NURSE_FEATURES, dtype=np.float32)
        return df


__all__ = ["NurseTriageBridge"]

